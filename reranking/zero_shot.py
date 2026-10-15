from __future__ import annotations

import json
import hashlib
import re
import tomllib
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from safetensors import safe_open

from retrieval.evaluation import ranking_metrics
from retrieval.io import read_dataframe, write_dataframe, write_json


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "reranker.toml"
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
CACHE_TEXT_COLUMNS = (
    "query_id",
    "article_id",
    "query_text",
    "article_title",
    "bm25_chunk_text",
    "dense_text",
)
REGRESSION_COLUMNS = (
    "query_id",
    "query_text",
    "ground_truth",
    "baseline_ap_at_10",
    "reranked_ap_at_10",
    "ap_delta",
    "category",
    "reason",
    "relevant_rank_changes",
    "promoted_article_id",
    "promoted_article_title",
    "demoted_relevant_id",
    "demoted_relevant_title",
    "article_similarity",
    "title_similarity",
    "baseline_top_10",
    "reranked_top_10",
)


@dataclass(frozen=True, slots=True)
class Config:
    oof_candidates: Path
    calibration: Path
    output_dir: Path
    model_id: str
    revision: str
    max_parameters: int
    max_length: int
    batch_size_gpu: int
    batch_size_cpu: int
    local_files_only: bool
    trust_remote_code: bool
    remote_code_repo: str | None
    remote_code_revision: str | None
    depths: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    current_best_depth: int | None
    current_best_fold_alphas: tuple[float, ...]


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    with path.open("rb") as source:
        raw = tomllib.load(source)
    base = path.parent.parent.resolve()

    def project(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else base / candidate

    paths = raw["paths"]
    model = raw["model"]
    evaluation = raw["evaluation"]
    config = Config(
        oof_candidates=project(paths["oof_candidates"]),
        calibration=project(paths["calibration"]),
        output_dir=project(paths["output_dir"]),
        model_id=str(model["model_id"]),
        revision=str(model["revision"]),
        max_parameters=int(model["max_parameters"]),
        max_length=int(model["max_length"]),
        batch_size_gpu=int(model["batch_size_gpu"]),
        batch_size_cpu=int(model["batch_size_cpu"]),
        local_files_only=bool(model["local_files_only"]),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        remote_code_repo=(
            str(model["remote_code_repo"])
            if model.get("remote_code_repo")
            else None
        ),
        remote_code_revision=(
            str(model["remote_code_revision"])
            if model.get("remote_code_revision")
            else None
        ),
        depths=tuple(int(value) for value in evaluation["depths"]),
        alpha_grid=tuple(float(value) for value in evaluation["alpha_grid"]),
        current_best_depth=(
            int(evaluation["current_best_depth"])
            if "current_best_depth" in evaluation
            else None
        ),
        current_best_fold_alphas=tuple(
            float(value) for value in evaluation.get("current_best_fold_alphas", [])
        ),
    )
    if config.max_parameters > 1_000_000_000:
        raise ValueError("configured parameter limit must not exceed 1B")
    if config.trust_remote_code and not (
        config.remote_code_repo and config.remote_code_revision
    ):
        raise ValueError("trusted remote code requires pinned repo and revision")
    if any(depth <= 0 or depth > 50 for depth in config.depths):
        raise ValueError("rerank depths must be in [1, 50]")
    if any(not 0.0 <= alpha <= 1.0 for alpha in config.alpha_grid):
        raise ValueError("alpha values must be in [0, 1]")
    if config.current_best_depth is not None and not config.current_best_fold_alphas:
        raise ValueError("current_best_fold_alphas are required with current_best_depth")
    return config


def cross_encoder_cache_identity(
    candidates: pd.DataFrame,
    config: Config,
) -> dict[str, object]:
    missing = set(CACHE_TEXT_COLUMNS) - set(candidates.columns)
    if missing:
        raise ValueError(f"missing cache identity columns: {sorted(missing)}")
    digest = hashlib.sha256()
    ordered = candidates.sort_values(
        ["query_id", "article_id"], kind="stable", ignore_index=True
    )
    for row in ordered[list(CACHE_TEXT_COLUMNS)].itertuples(index=False, name=None):
        for value in row:
            encoded = ("" if pd.isna(value) else str(value)).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return {
        "schema_version": 2,
        "model_id": config.model_id,
        "revision": config.revision,
        "remote_code_repo": (
            config.remote_code_repo if config.trust_remote_code else None
        ),
        "remote_code_revision": (
            config.remote_code_revision if config.trust_remote_code else None
        ),
        "max_length": config.max_length,
        "view_schema_version": 1,
        "candidate_count": len(ordered),
        "candidate_text_sha256": digest.hexdigest(),
    }


def score_cache_matches(
    manifest_path: Path,
    expected_identity: dict[str, object],
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("cache_identity") == expected_identity


def validate_score_cache(scores: pd.DataFrame, candidates: pd.DataFrame) -> None:
    required = {"query_id", "article_id", "cross_encoder_score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"missing cross-encoder score columns: {sorted(missing)}")
    if scores.duplicated(["query_id", "article_id"]).any():
        raise ValueError("cross-encoder score cache contains duplicate pairs")
    expected = set(
        zip(candidates["query_id"].astype(int), candidates["article_id"].astype(int))
    )
    actual = set(zip(scores["query_id"].astype(int), scores["article_id"].astype(int)))
    if actual != expected:
        raise ValueError("cross-encoder score cache candidate pairs do not match")
    if not np.isfinite(pd.to_numeric(scores["cross_encoder_score"], errors="coerce")).all():
        raise ValueError("cross-encoder score cache contains non-finite values")


def evaluate_zero_shot(config: Config) -> dict[str, object]:
    candidates = read_dataframe(config.oof_candidates)
    calibration = read_dataframe(config.calibration)
    _validate_candidates(candidates)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = ranking_metrics(candidates, calibration)
    score_path = config.output_dir / "cross_encoder_scores.f"
    manifest_path = config.output_dir / "model_manifest.json"
    cache_identity = cross_encoder_cache_identity(candidates, config)
    if score_path.exists() and score_cache_matches(manifest_path, cache_identity):
        scores = read_dataframe(score_path)
        validate_score_cache(scores, candidates)
    else:
        scores, parameter_count, device = score_cross_encoder(candidates, config)
        validate_score_cache(scores, candidates)
        write_dataframe(scores, score_path)
        write_json(
            {
                "cache_identity": cache_identity,
                "model_id": config.model_id,
                "revision": config.revision,
                "parameter_count": parameter_count,
                "device": device,
            },
            manifest_path,
        )
    scored = candidates.merge(scores, on=["query_id", "article_id"], how="left")
    if scored["cross_encoder_score"].isna().any():
        raise ValueError("cross-encoder score cache does not cover all candidates")

    experiment_rows: list[dict[str, object]] = [
        {"experiment": "linear_fusion", "depth": 0, "alpha": 0.0, **baseline}
    ]
    fold_rows: list[dict[str, object]] = []
    rankings: list[pd.DataFrame] = []
    for depth in config.depths:
        nested_ranked, nested_metrics, nested_folds = nested_alpha_evaluation(
            scored,
            calibration,
            depth=depth,
            alpha_grid=config.alpha_grid,
        )
        experiment_rows.append(
            {
                "experiment": "zero_shot_nested_oof",
                "depth": depth,
                "alpha": None,
                **nested_metrics,
            }
        )
        fold_rows.extend(nested_folds)
        nested_ranked["experiment_depth"] = depth
        rankings.append(nested_ranked)

    metrics = pd.DataFrame(experiment_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    nested_rows = metrics[metrics["experiment"] == "zero_shot_nested_oof"]
    best_row = nested_rows.sort_values(
        ["map_at_10", "recall_at_10", "depth"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    best_depth = int(best_row["depth"])
    fold_signal = _fold_signal_summary(fold_metrics)
    metrics.to_csv(config.output_dir / "metrics.csv", index=False)
    fold_metrics.to_csv(config.output_dir / "fold_metrics.csv", index=False)
    combined_rankings = pd.concat(rankings, ignore_index=True)
    write_dataframe(combined_rankings, config.output_dir / "oof_rankings.f")

    regression_summaries: dict[str, object] = {}
    for depth in sorted({30, best_depth} & set(config.depths)):
        depth_ranked = combined_rankings[
            combined_rankings["experiment_depth"] == depth
        ]
        regressions = analyze_regressions(scored, depth_ranked, calibration)
        regressions.to_csv(
            config.output_dir / f"regressions_depth_{depth}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary = regression_summary(regressions)
        regression_summaries[str(depth)] = summary
        write_json(summary, config.output_dir / f"regressions_depth_{depth}.json")

    if config.current_best_depth is not None:
        folds = sorted(scored["fold"].astype(int).unique())
        if len(folds) != len(config.current_best_fold_alphas):
            raise ValueError("current_best_fold_alphas must match the outer fold count")
        fixed_alphas = dict(zip(folds, config.current_best_fold_alphas))
        current_best_parts = [
            rerank_candidates(
                scored[scored["fold"] == fold],
                depth=config.current_best_depth,
                alpha=fixed_alphas[fold],
            )
            for fold in folds
        ]
        current_best_ranked = pd.concat(current_best_parts, ignore_index=True).sort_values(
            ["query_id", "candidate_rank"], ignore_index=True
        )
        current_best_metrics = ranking_metrics(current_best_ranked, calibration)
        current_best_regressions = analyze_regressions(
            scored, current_best_ranked, calibration
        )
        write_dataframe(
            current_best_ranked,
            config.output_dir / "current_best_oof_rankings.f",
        )
        current_best_regressions.to_csv(
            config.output_dir
            / f"regressions_current_best_depth_{config.current_best_depth}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        current_best_summary = regression_summary(current_best_regressions)
        regression_summaries[
            f"current_best_depth_{config.current_best_depth}"
        ] = current_best_summary
        write_json(
            {
                "status": "current_best_zero_shot_baseline",
                "model_id": config.model_id,
                "revision": config.revision,
                "method": (
                    f"GTE cross-encoder top-{config.current_best_depth} "
                    "+ tuned linear fusion"
                ),
                "depth": config.current_best_depth,
                "alpha_selection": "nested OOF selection",
                "alpha_by_outer_fold": {
                    str(fold): fixed_alphas[fold] for fold in folds
                },
                **current_best_metrics,
                "baseline_map_at_10": float(baseline["map_at_10"]),
                "regressions": current_best_summary,
            },
            config.output_dir / "current_best_zero_shot.json",
        )
    write_json(
        {
            "selection_scope": "nested OOF depth comparison",
            "best_depth": best_depth,
            "map_at_10": float(best_row["map_at_10"]),
            "recall_at_10": float(best_row["recall_at_10"]),
            "fold_signal": fold_signal[str(best_depth)],
        },
        config.output_dir / "depth_search_best.json",
    )
    experiment_payload = [
        {
            key: None if pd.isna(value) else value
            for key, value in row.items()
        }
        for row in metrics.to_dict("records")
    ]
    payload = {
        "baseline": baseline,
        "experiments": experiment_payload,
        "best_depth": best_depth,
        "fold_signal": fold_signal,
        "regressions": regression_summaries,
    }
    write_json(payload, config.output_dir / "metrics.json")
    return payload


def score_cross_encoder(
    candidates: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, int, str]:
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder

    snapshot = Path(
        snapshot_download(
            repo_id=config.model_id,
            revision=config.revision,
            allow_patterns=list(MODEL_FILES),
            local_files_only=config.local_files_only,
        )
    )
    parameter_count = count_safetensor_parameters(snapshot)
    enforce_parameter_limit(parameter_count, config.max_parameters)
    if config.trust_remote_code:
        remote_code_path = download_remote_code(
            snapshot_download, config
        )
        if remote_code_path.name != config.remote_code_revision:
            raise ValueError(
                "Alibaba remote-code revision changed; update the pinned revision explicitly"
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(
        str(snapshot),
        max_length=config.max_length,
        device=device,
        trust_remote_code=config.trust_remote_code,
        local_files_only=True,
        model_kwargs=cross_encoder_model_kwargs(config),
    )
    if config.trust_remote_code:
        _repair_gte_nonpersistent_buffers(model.model)
    pairs, candidate_indices, sources = build_cross_encoder_views(candidates)
    raw_scores = model.predict(
        pairs,
        batch_size=config.batch_size_gpu if device == "cuda" else config.batch_size_cpu,
        show_progress_bar=True,
        activation_fn=torch.nn.Identity(),
        convert_to_numpy=True,
    )
    raw_scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    best_scores = np.full(len(candidates), -np.inf, dtype=np.float32)
    best_sources = np.full(len(candidates), "", dtype=object)
    for score, candidate_index, source in zip(raw_scores, candidate_indices, sources):
        if score > best_scores[candidate_index]:
            best_scores[candidate_index] = score
            best_sources[candidate_index] = source
    if not np.isfinite(best_scores).all():
        raise RuntimeError("at least one candidate has no cross-encoder view")
    return (
        pd.DataFrame(
            {
                "query_id": candidates["query_id"].astype(int),
                "article_id": candidates["article_id"].astype(int),
                "cross_encoder_score": best_scores,
                "best_chunk_source": best_sources,
            }
        ),
        parameter_count,
        device,
    )


def cross_encoder_model_kwargs(config: Config) -> dict[str, object]:
    kwargs: dict[str, object] = {"use_safetensors": True, "dtype": "auto"}
    if config.trust_remote_code:
        kwargs["code_revision"] = config.remote_code_revision
    return kwargs


def download_remote_code(snapshot_download: object, config: Config) -> Path:
    if not config.remote_code_repo or not config.remote_code_revision:
        raise ValueError("remote code repo and revision are required")
    return Path(
        snapshot_download(
            repo_id=config.remote_code_repo,
            revision=config.remote_code_revision,
            allow_patterns=["configuration.py", "modeling.py"],
            local_files_only=config.local_files_only,
        )
    )


def build_cross_encoder_views(
    candidates: pd.DataFrame,
) -> tuple[list[tuple[str, str]], np.ndarray, list[str]]:
    pairs: list[tuple[str, str]] = []
    candidate_indices: list[int] = []
    sources: list[str] = []
    for candidate_index, row in enumerate(candidates.itertuples(index=False)):
        title = str(row.article_title).strip()
        chunks = [
            ("bm25", str(getattr(row, "bm25_chunk_text", "") or "").strip()),
            ("dense", str(getattr(row, "dense_text", "") or "").strip()),
        ]
        seen: set[str] = set()
        views = []
        for source, chunk in chunks:
            key = _text_key(chunk)
            if chunk and key not in seen:
                seen.add(key)
                views.append((source, chunk))
        if not views:
            views = [("title", "")]
        for source, chunk in views:
            document = f"Заголовок: {title}"
            if chunk:
                document += f"\nФрагмент: {chunk}"
            pairs.append((str(row.query_text), document))
            candidate_indices.append(candidate_index)
            sources.append(source)
    return pairs, np.asarray(candidate_indices, dtype=np.int32), sources


def rerank_candidates(
    candidates: pd.DataFrame,
    depth: int,
    alpha: float,
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for _, group in candidates.groupby("query_id", sort=False):
        base = group.sort_values("candidate_rank", kind="stable").copy()
        active = base[base["candidate_rank"] <= depth].copy()
        tail = base[base["candidate_rank"] > depth].copy()
        active["_ce_normalized"] = _minmax(active["cross_encoder_score"])
        active["_fusion_normalized"] = _minmax(active["fusion_score"])
        active["final_score"] = (
            float(alpha) * active["_ce_normalized"]
            + (1.0 - float(alpha)) * active["_fusion_normalized"]
        )
        active = active.sort_values(
            ["final_score", "candidate_rank", "article_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        tail["final_score"] = -np.inf
        ranked = pd.concat([active, tail], ignore_index=True)
        ranked["candidate_rank"] = range(1, len(ranked) + 1)
        output.append(ranked)
    return pd.concat(output, ignore_index=True)


def tune_alpha(
    candidates: pd.DataFrame,
    calibration: pd.DataFrame,
    depth: int,
    alpha_grid: Sequence[float],
) -> tuple[float, dict[str, float]]:
    best_alpha = float(alpha_grid[0])
    best_metrics = {"map_at_10": -1.0, "recall_at_10": -1.0}
    for alpha in alpha_grid:
        metrics = fast_ranking_metrics(candidates, calibration, depth, float(alpha))
        key = (metrics["map_at_10"], metrics["recall_at_10"], -float(alpha))
        best_key = (
            best_metrics["map_at_10"],
            best_metrics["recall_at_10"],
            -best_alpha,
        )
        if key > best_key:
            best_alpha = float(alpha)
            best_metrics = metrics
    return best_alpha, best_metrics


def fast_ranking_metrics(
    candidates: pd.DataFrame,
    calibration: pd.DataFrame,
    depth: int,
    alpha: float,
) -> dict[str, float]:
    ordered = candidates.sort_values(
        ["query_id", "candidate_rank"], kind="stable", ignore_index=True
    )
    query_ids = ordered["query_id"].drop_duplicates().to_numpy(dtype=np.int64)
    per_query = len(ordered) // len(query_ids)
    if per_query * len(query_ids) != len(ordered):
        raise ValueError("candidate matrix is not rectangular")
    shape = (len(query_ids), per_query)
    cross_encoder = ordered["cross_encoder_score"].to_numpy(dtype=np.float64).reshape(shape)
    fusion = ordered["fusion_score"].to_numpy(dtype=np.float64).reshape(shape)
    article_ids = ordered["article_id"].to_numpy(dtype=np.int64).reshape(shape)
    cross_encoder = _row_minmax(cross_encoder[:, :depth])
    fusion = _row_minmax(fusion[:, :depth])
    scores = float(alpha) * cross_encoder + (1.0 - float(alpha)) * fusion
    active_limit = min(depth, 10)
    ranking = np.argsort(-scores, axis=1, kind="stable")[:, :active_limit]
    ranked_active = np.take_along_axis(article_ids[:, :depth], ranking, axis=1)
    if depth < 10:
        unchanged_tail = article_ids[:, depth : min(per_query, 10)]
        ranked_articles = np.concatenate([ranked_active, unchanged_tail], axis=1)
    else:
        ranked_articles = ranked_active
    truth = {
        int(row.query_id): {int(value) for value in str(row.ground_truth).split()}
        for row in calibration.itertuples(index=False)
    }
    average_precision: list[float] = []
    recalls: list[float] = []
    for row_index, query_id in enumerate(query_ids):
        relevant = truth[int(query_id)]
        hits = 0
        precision_sum = 0.0
        seen: set[int] = set()
        for rank, article_id in enumerate(ranked_articles[row_index], start=1):
            article_id = int(article_id)
            if article_id in relevant and article_id not in seen:
                hits += 1
                precision_sum += hits / rank
            seen.add(article_id)
        average_precision.append(precision_sum / min(len(relevant), 10))
        recalls.append(hits / len(relevant))
    return {
        "map_at_10": round(float(np.mean(average_precision)), 6),
        "recall_at_10": round(float(np.mean(recalls)), 6),
    }


def nested_alpha_evaluation(
    candidates: pd.DataFrame,
    calibration: pd.DataFrame,
    depth: int,
    alpha_grid: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, float], list[dict[str, object]]]:
    ranked_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for fold in sorted(candidates["fold"].astype(int).unique()):
        training = candidates[candidates["fold"] != fold]
        validation = candidates[candidates["fold"] == fold]
        training_truth = calibration[
            calibration["query_id"].isin(training["query_id"].astype(int).unique())
        ]
        validation_truth = calibration[
            calibration["query_id"].isin(validation["query_id"].astype(int).unique())
        ]
        alpha, train_metrics = tune_alpha(
            training, training_truth, depth=depth, alpha_grid=alpha_grid
        )
        ranked = rerank_candidates(validation, depth=depth, alpha=alpha)
        validation_metrics = ranking_metrics(ranked, validation_truth)
        baseline_metrics = ranking_metrics(validation, validation_truth)
        ranked_parts.append(ranked)
        fold_rows.append(
            {
                "depth": depth,
                "fold": int(fold),
                "alpha": alpha,
                "train_map_at_10": train_metrics["map_at_10"],
                "baseline_map_at_10": baseline_metrics["map_at_10"],
                "map_at_10_delta": round(
                    validation_metrics["map_at_10"] - baseline_metrics["map_at_10"],
                    6,
                ),
                "map_at_10_improved": (
                    validation_metrics["map_at_10"] > baseline_metrics["map_at_10"]
                ),
                **validation_metrics,
            }
        )
    combined = pd.concat(ranked_parts, ignore_index=True).sort_values(
        ["query_id", "candidate_rank"], ignore_index=True
    )
    return combined, ranking_metrics(combined, calibration), fold_rows


def analyze_regressions(
    baseline: pd.DataFrame,
    reranked: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Explain per-query AP regressions using observable rank and text evidence."""
    truth_by_query = {
        int(row.query_id): [int(value) for value in str(row.ground_truth).split()]
        for row in calibration.itertuples(index=False)
    }
    query_text = {
        int(row.query_id): str(row.query_text)
        for row in calibration.itertuples(index=False)
    }
    reranked_groups = {
        int(query_id): group.sort_values("candidate_rank", kind="stable")
        for query_id, group in reranked.groupby("query_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    for query_id, source in baseline.groupby("query_id", sort=False):
        query_id = int(query_id)
        source = source.sort_values("candidate_rank", kind="stable")
        ranked = reranked_groups[query_id]
        relevant = set(truth_by_query[query_id])
        baseline_ids = source["article_id"].astype(int).tolist()
        reranked_ids = ranked["article_id"].astype(int).tolist()
        baseline_ap = _average_precision_at_10(baseline_ids, relevant)
        reranked_ap = _average_precision_at_10(reranked_ids, relevant)
        if reranked_ap >= baseline_ap:
            continue

        baseline_ranks = {article_id: rank for rank, article_id in enumerate(baseline_ids, 1)}
        reranked_ranks = {article_id: rank for rank, article_id in enumerate(reranked_ids, 1)}
        regressed_relevant = sorted(
            relevant,
            key=lambda article_id: (
                reranked_ranks.get(article_id, 10**6)
                - baseline_ranks.get(article_id, 10**6)
            ),
            reverse=True,
        )
        regressed_relevant = [
            article_id
            for article_id in regressed_relevant
            if reranked_ranks.get(article_id, 10**6)
            > baseline_ranks.get(article_id, 10**6)
        ]
        promoted = [
            article_id
            for article_id in reranked_ids[:10]
            if article_id not in relevant
            and reranked_ranks[article_id] < baseline_ranks.get(article_id, 10**6)
        ]
        promoted.sort(
            key=lambda article_id: baseline_ranks.get(article_id, 10**6)
            - reranked_ranks[article_id],
            reverse=True,
        )
        confused_id = promoted[0] if promoted else next(
            (article_id for article_id in reranked_ids[:10] if article_id not in relevant),
            None,
        )
        lost_id = regressed_relevant[0] if regressed_relevant else next(iter(relevant))
        source_by_id = source.set_index(source["article_id"].astype(int), drop=False)
        similarity = _article_similarity(source_by_id, lost_id, confused_id)
        title_similarity = _title_similarity(source_by_id, lost_id, confused_id)
        lost_second = (
            len(relevant) > 1
            and any(
                baseline_ranks.get(article_id, 10**6)
                < reranked_ranks.get(article_id, 10**6)
                for article_id in relevant
            )
            and any(reranked_ranks.get(article_id, 10**6) <= 10 for article_id in relevant)
        )
        if lost_second:
            category = "loses_second_relevant"
            reason = (
                f"Relevant {lost_id} moved "
                f"{baseline_ranks.get(lost_id, 0)}->{reranked_ranks.get(lost_id, 0)} "
                "while another relevant article remained in top-10."
            )
        elif similarity >= 0.30 or title_similarity >= 0.45:
            category = "confuses_close_articles"
            reason = (
                f"Non-relevant {confused_id} was promoted over relevant {lost_id}; "
                f"text similarity={similarity:.3f}, title similarity={title_similarity:.3f}."
            )
        else:
            category = "overweights_general_topic"
            reason = (
                f"Non-relevant {confused_id} was promoted despite low direct similarity "
                f"to relevant {lost_id} ({similarity:.3f})."
            )

        rows.append(
            {
                "query_id": query_id,
                "query_text": query_text[query_id],
                "ground_truth": " ".join(map(str, truth_by_query[query_id])),
                "baseline_ap_at_10": round(baseline_ap, 6),
                "reranked_ap_at_10": round(reranked_ap, 6),
                "ap_delta": round(reranked_ap - baseline_ap, 6),
                "category": category,
                "reason": reason,
                "relevant_rank_changes": _format_rank_changes(
                    truth_by_query[query_id], baseline_ranks, reranked_ranks
                ),
                "promoted_article_id": confused_id,
                "promoted_article_title": _article_title(source_by_id, confused_id),
                "demoted_relevant_id": lost_id,
                "demoted_relevant_title": _article_title(source_by_id, lost_id),
                "article_similarity": round(similarity, 6),
                "title_similarity": round(title_similarity, 6),
                "baseline_top_10": " ".join(map(str, baseline_ids[:10])),
                "reranked_top_10": " ".join(map(str, reranked_ids[:10])),
            }
        )
    if not rows:
        return pd.DataFrame(columns=REGRESSION_COLUMNS)
    return pd.DataFrame(rows, columns=REGRESSION_COLUMNS).sort_values(
        ["ap_delta", "query_id"], kind="stable", ignore_index=True
    )


def regression_summary(regressions: pd.DataFrame) -> dict[str, object]:
    counts = Counter(regressions.get("category", pd.Series(dtype=str)).astype(str))
    return {
        "analyzed_query_count": int(len(regressions)),
        "category_counts": dict(sorted(counts.items())),
        "mean_ap_delta": round(float(regressions["ap_delta"].mean()), 6)
        if len(regressions)
        else 0.0,
    }


def _fold_signal_summary(fold_metrics: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for depth, rows in fold_metrics.groupby("depth", sort=True):
        improved = int(rows["map_at_10_improved"].sum())
        result[str(int(depth))] = {
            "improved_folds": improved,
            "total_folds": int(len(rows)),
            "good_signal": improved >= 4,
        }
    return result


def _average_precision_at_10(ranking: Sequence[int], relevant: set[int]) -> float:
    hits = 0
    precision_sum = 0.0
    seen: set[int] = set()
    for rank, article_id in enumerate(ranking[:10], start=1):
        article_id = int(article_id)
        if article_id in relevant and article_id not in seen:
            hits += 1
            precision_sum += hits / rank
        seen.add(article_id)
    return precision_sum / min(len(relevant), 10)


def _format_rank_changes(
    article_ids: Sequence[int],
    baseline_ranks: dict[int, int],
    reranked_ranks: dict[int, int],
) -> str:
    return "; ".join(
        f"{article_id}:{baseline_ranks.get(article_id, 0)}->{reranked_ranks.get(article_id, 0)}"
        for article_id in article_ids
    )


def _article_title(source_by_id: pd.DataFrame, article_id: int | None) -> str:
    if article_id is None or article_id not in source_by_id.index:
        return ""
    return str(source_by_id.loc[article_id, "article_title"])


def _article_similarity(
    source_by_id: pd.DataFrame,
    relevant_id: int,
    promoted_id: int | None,
) -> float:
    if promoted_id is None or relevant_id not in source_by_id.index or promoted_id not in source_by_id.index:
        return 0.0

    def tokens(article_id: int) -> set[str]:
        row = source_by_id.loc[article_id]
        value = " ".join(
            str(row.get(column, "") or "")
            for column in ("article_title", "bm25_chunk_text", "dense_text")
        ).casefold()
        return set(re.findall(r"\w+", value, flags=re.UNICODE))

    relevant_tokens = tokens(relevant_id)
    promoted_tokens = tokens(promoted_id)
    union = relevant_tokens | promoted_tokens
    return len(relevant_tokens & promoted_tokens) / len(union) if union else 0.0


def _title_similarity(
    source_by_id: pd.DataFrame,
    relevant_id: int,
    promoted_id: int | None,
) -> float:
    if promoted_id is None or relevant_id not in source_by_id.index or promoted_id not in source_by_id.index:
        return 0.0
    relevant = _text_key(str(source_by_id.loc[relevant_id, "article_title"]))
    promoted = _text_key(str(source_by_id.loc[promoted_id, "article_title"]))
    return SequenceMatcher(None, relevant, promoted).ratio()


def count_safetensor_parameters(model_dir: Path) -> int:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights found in {model_dir}")
    total = 0
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as source:
            for key in source.keys():
                total += int(np.prod(source.get_slice(key).get_shape(), dtype=np.int64))
    return total


def enforce_parameter_limit(parameter_count: int, limit: int) -> None:
    if parameter_count > limit:
        raise ValueError(
            f"Model has {parameter_count:,} parameters; limit is {limit:,}"
        )


def _repair_gte_nonpersistent_buffers(model: object) -> None:
    import torch

    embeddings = model.new.embeddings
    device = embeddings.word_embeddings.weight.device
    embeddings.position_ids = torch.arange(
        model.config.max_position_embeddings, device=device
    )
    rotary = embeddings.rotary_emb
    embeddings.rotary_emb = type(rotary)(
        dim=rotary.dim,
        max_position_embeddings=rotary.max_position_embeddings,
        base=rotary.base,
        scaling_factor=rotary.scaling_factor,
        mixed_b=rotary.mixed_b,
    ).to(device)


def _validate_candidates(candidates: pd.DataFrame) -> None:
    required = {
        "query_id",
        "query_text",
        "article_id",
        "candidate_rank",
        "article_title",
        "fusion_score",
        "fold",
        "bm25_chunk_text",
        "dense_text",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"missing candidate columns: {sorted(missing)}")
    if candidates.duplicated(["query_id", "article_id"]).any():
        raise ValueError("candidate pairs must be unique")
    if not candidates.groupby("query_id").size().eq(50).all():
        raise ValueError("every query must contain 50 candidates")


def _minmax(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    minimum = float(array.min())
    maximum = float(array.max())
    return (array - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(array)


def _row_minmax(values: np.ndarray) -> np.ndarray:
    minima = values.min(axis=1, keepdims=True)
    spans = values.max(axis=1, keepdims=True) - minima
    return np.divide(
        values - minima,
        spans,
        out=np.zeros_like(values),
        where=spans > 0,
    )


def _text_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()
