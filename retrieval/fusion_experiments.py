from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from retrieval.channels import QueryKnnChannel
from retrieval.config import CandidateGenerationConfig
from retrieval.dense import DenseEncoder
from retrieval.evaluation import grouped_multilabel_folds
from retrieval.index import UniversalIndex
from retrieval.io import write_dataframe, write_json


CHANNELS = ("bm25_title", "bm25_chunk", "char_tfidf", "dense", "query_knn")
NORMALIZATIONS = ("max", "minmax")
WEIGHT_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


@dataclass(slots=True)
class FeatureCube:
    frame: pd.DataFrame
    query_ids: np.ndarray
    article_ids: np.ndarray
    folds: np.ndarray
    relevant: np.ndarray
    ranks: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]

    @property
    def query_count(self) -> int:
        return len(self.query_ids)

    @property
    def article_count(self) -> int:
        return len(self.article_ids)


@dataclass(slots=True)
class FusionExperimentResult:
    comparison: pd.DataFrame
    meta_folds: pd.DataFrame
    ablations: pd.DataFrame
    best_method: str
    best_normalization: str | None
    final_weights: dict[str, float]
    meta_candidates: pd.DataFrame
    production_oof_candidates: pd.DataFrame


def build_oof_features(
    index: UniversalIndex,
    encoder: DenseEncoder,
    calibration: pd.DataFrame,
    config: CandidateGenerationConfig,
) -> pd.DataFrame:
    folds = grouped_multilabel_folds(
        calibration,
        n_splits=config.evaluation.folds,
        seed=config.evaluation.seed,
    )
    universal_channels = index.channels(encoder)
    article_ids = np.sort(index.articles["article_id"].astype(int).unique())
    parts: list[pd.DataFrame] = []
    for fold_index, (train_indices, validation_indices) in enumerate(folds):
        train = calibration.iloc[train_indices].reset_index(drop=True)
        validation = calibration.iloc[validation_indices].reset_index(drop=True)
        query_knn = QueryKnnChannel(
            neighbors=config.retrieval.knn_neighbors,
            frequency_penalty=config.retrieval.knn_frequency_penalty,
            word_weight=config.retrieval.knn_word_weight,
            voting=config.retrieval.knn_voting,
        ).fit(train)
        channels = {**universal_channels, "query_knn": query_knn}
        results = {
            name: channel.search(validation, len(article_ids))
            for name, channel in channels.items()
        }
        fold_frame = _complete_feature_grid(validation, article_ids, results)
        fold_frame["fold"] = fold_index
        truth = {
            int(row.query_id): {int(value) for value in str(row.ground_truth).split()}
            for row in validation.itertuples(index=False)
        }
        fold_frame["is_relevant"] = [
            int(article_id) in truth[int(query_id)]
            for query_id, article_id in zip(
                fold_frame["query_id"], fold_frame["article_id"]
            )
        ]
        parts.append(fold_frame)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["query_id", "article_id"], kind="stable", ignore_index=True
    )


def build_inference_features(
    index: UniversalIndex,
    encoder: DenseEncoder,
    calibration: pd.DataFrame,
    queries: pd.DataFrame,
    config: CandidateGenerationConfig,
) -> pd.DataFrame:
    article_ids = np.sort(index.articles["article_id"].astype(int).unique())
    query_knn = QueryKnnChannel(
        neighbors=config.retrieval.knn_neighbors,
        frequency_penalty=config.retrieval.knn_frequency_penalty,
        word_weight=config.retrieval.knn_word_weight,
        voting=config.retrieval.knn_voting,
    ).fit(calibration)
    channels = {**index.channels(encoder), "query_knn": query_knn}
    results = {
        name: channel.search(queries, len(article_ids))
        for name, channel in channels.items()
    }
    return _complete_feature_grid(queries, article_ids, results).sort_values(
        ["query_id", "article_id"], kind="stable", ignore_index=True
    )


def _complete_feature_grid(
    queries: pd.DataFrame,
    article_ids: np.ndarray,
    channel_results: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [queries["query_id"].astype(int), article_ids],
        names=["query_id", "article_id"],
    ).to_frame(index=False)
    for name in CHANNELS:
        result = channel_results[name]
        columns = ["query_id", "article_id", "score", "rank"]
        if name in {"bm25_chunk", "dense"}:
            columns.append("best_chunk_id")
        selected = result[columns].rename(
            columns={
                "score": f"{name}_score",
                "rank": f"{name}_rank",
                "best_chunk_id": f"best_{name}_chunk_id",
            }
        )
        grid = grid.merge(selected, on=["query_id", "article_id"], how="left")
    for name in CHANNELS:
        grid[f"{name}_score"] = grid[f"{name}_score"].fillna(0.0).astype(np.float32)
        grid[f"{name}_rank"] = grid[f"{name}_rank"].astype("Int32")
    return grid


def make_feature_cube(features: pd.DataFrame) -> FeatureCube:
    ordered = features.sort_values(
        ["query_id", "article_id"], kind="stable", ignore_index=True
    )
    query_ids = ordered["query_id"].drop_duplicates().to_numpy(dtype=np.int64)
    article_ids = ordered["article_id"].drop_duplicates().to_numpy(dtype=np.int64)
    expected = len(query_ids) * len(article_ids)
    if len(ordered) != expected:
        raise ValueError("feature matrix must contain every query/article pair")
    shape = (len(query_ids), len(article_ids))
    folds = (
        ordered.groupby("query_id", sort=False)["fold"].first().to_numpy(dtype=np.int16)
        if "fold" in ordered
        else np.full(len(query_ids), -1, dtype=np.int16)
    )
    relevant = (
        ordered["is_relevant"].to_numpy(dtype=bool).reshape(shape)
        if "is_relevant" in ordered
        else np.zeros(shape, dtype=bool)
    )
    ranks = {
        name: ordered[f"{name}_rank"].fillna(0).to_numpy(dtype=np.float32).reshape(shape)
        for name in CHANNELS
    }
    scores = {
        name: ordered[f"{name}_score"].to_numpy(dtype=np.float32).reshape(shape)
        for name in CHANNELS
    }
    return FeatureCube(
        frame=ordered,
        query_ids=query_ids,
        article_ids=article_ids,
        folds=folds,
        relevant=relevant,
        ranks=ranks,
        scores=scores,
    )


def contribution_tensor(
    cube: FeatureCube,
    method: str,
    rrf_k: int = 60,
    normalization: str | None = None,
) -> np.ndarray:
    if method == "rrf":
        return np.stack(
            [
                np.where(cube.ranks[name] > 0, 1.0 / (rrf_k + cube.ranks[name]), 0.0)
                for name in CHANNELS
            ],
            axis=0,
        ).astype(np.float32)
    if method != "linear":
        raise ValueError(f"unknown fusion method: {method}")
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"unknown score normalization: {normalization}")
    return np.stack(
        [normalize_channel_scores(cube.scores[name], cube.ranks[name], normalization) for name in CHANNELS],
        axis=0,
    )


def normalize_channel_scores(
    scores: np.ndarray,
    ranks: np.ndarray,
    method: str,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    present = np.asarray(ranks) > 0
    result = np.zeros_like(values)
    if method == "max":
        maxima = np.max(np.where(present, values, 0.0), axis=1, keepdims=True)
        np.divide(values, maxima, out=result, where=present & (maxima > 0))
        return result
    if method == "minmax":
        minima = np.min(np.where(present, values, np.inf), axis=1, keepdims=True)
        maxima = np.max(np.where(present, values, -np.inf), axis=1, keepdims=True)
        spans = maxima - minima
        np.divide(values - minima, spans, out=result, where=present & (spans > 0))
        result[present & np.broadcast_to(spans <= 0, result.shape)] = 1.0
        return result
    raise ValueError(f"unknown score normalization: {method}")


def ranking_metrics_from_scores(
    scores: np.ndarray,
    relevant: np.ndarray,
    query_mask: np.ndarray | None = None,
) -> dict[str, float]:
    if query_mask is not None:
        scores = scores[query_mask]
        relevant = relevant[query_mask]
    order = np.argsort(-scores, axis=1, kind="stable")
    relevant_count = relevant.sum(axis=1)
    if np.any(relevant_count == 0):
        raise ValueError("every evaluated query must have at least one relevant article")
    map_cutoff = min(10, scores.shape[1])
    top10 = np.take_along_axis(relevant, order[:, :map_cutoff], axis=1)
    cumulative = np.cumsum(top10, axis=1)
    precision = cumulative / np.arange(1, map_cutoff + 1, dtype=np.float32)
    average_precision = (precision * top10).sum(axis=1) / np.minimum(relevant_count, 10)
    result = {"map_at_10": round(float(average_precision.mean()), 6)}
    for cutoff in (10, 20, 50, 100):
        hits = np.take_along_axis(relevant, order[:, :cutoff], axis=1).sum(axis=1)
        result[f"recall_at_{cutoff}"] = round(float(np.mean(hits / relevant_count)), 6)
    return result


def weighted_scores(contributions: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    return np.tensordot(np.asarray(weights, dtype=np.float32), contributions, axes=(0, 0))


def tune_weights(
    contributions: np.ndarray,
    relevant: np.ndarray,
    query_mask: np.ndarray,
    minimum_recall: float,
) -> tuple[np.ndarray, dict[str, float]]:
    starts = [
        np.ones(len(CHANNELS), dtype=np.float32),
        np.asarray([0.25, 1.0, 1.0, 1.0, 3.0], dtype=np.float32),
        np.asarray([0.1, 0.5, 0.5, 0.5, 4.0], dtype=np.float32),
    ]
    best_weights = starts[0]
    best_metrics = ranking_metrics_from_scores(
        weighted_scores(contributions, best_weights), relevant, query_mask
    )
    for start in starts:
        current = start.copy()
        current_metrics = ranking_metrics_from_scores(
            weighted_scores(contributions, current), relevant, query_mask
        )
        for _ in range(3):
            changed = False
            for channel_index in range(len(CHANNELS)):
                local_weights = current
                local_metrics = current_metrics
                for value in WEIGHT_GRID:
                    candidate = current.copy()
                    candidate[channel_index] = value
                    if not candidate.any():
                        continue
                    metrics = ranking_metrics_from_scores(
                        weighted_scores(contributions, candidate), relevant, query_mask
                    )
                    if _metric_key(metrics, minimum_recall) > _metric_key(
                        local_metrics, minimum_recall
                    ):
                        local_weights = candidate
                        local_metrics = metrics
                if not np.array_equal(local_weights, current):
                    current = local_weights
                    current_metrics = local_metrics
                    changed = True
            if not changed:
                break
        if _metric_key(current_metrics, minimum_recall) > _metric_key(
            best_metrics, minimum_recall
        ):
            best_weights = current
            best_metrics = current_metrics
    scale = float(best_weights.max())
    if scale > 0:
        best_weights = best_weights / scale
    best_metrics = ranking_metrics_from_scores(
        weighted_scores(contributions, best_weights), relevant, query_mask
    )
    return best_weights, best_metrics


def _metric_key(metrics: Mapping[str, float], minimum_recall: float) -> tuple[float, ...]:
    recall = float(metrics["recall_at_100"])
    if recall >= minimum_recall:
        return (1.0, float(metrics["map_at_10"]), recall, float(metrics["recall_at_50"]))
    return (0.0, recall, float(metrics["map_at_10"]), float(metrics["recall_at_50"]))


def run_fusion_experiments(
    features: pd.DataFrame,
    config: CandidateGenerationConfig,
) -> FusionExperimentResult:
    cube = make_feature_cube(features)
    methods = (("rrf", None), ("linear", "max"), ("linear", "minmax"))
    equal_rrf = contribution_tensor(cube, "rrf", rrf_k=config.fusion.rrf_k)
    equal_metrics = ranking_metrics_from_scores(
        weighted_scores(equal_rrf, np.ones(len(CHANNELS))), cube.relevant
    )
    target_recall = max(0.0, equal_metrics["recall_at_100"] - 0.005)
    fold_rows: list[dict[str, object]] = []
    method_candidates: dict[tuple[str, str | None], list[pd.DataFrame]] = {
        method: [] for method in methods
    }
    for method, normalization in methods:
        contributions = contribution_tensor(
            cube,
            method,
            rrf_k=config.fusion.rrf_k,
            normalization=normalization,
        )
        for fold in sorted(np.unique(cube.folds)):
            train_mask = cube.folds != fold
            validation_mask = cube.folds == fold
            train_equal = ranking_metrics_from_scores(
                weighted_scores(contributions, np.ones(len(CHANNELS))),
                cube.relevant,
                train_mask,
            )
            fold_target = max(0.0, train_equal["recall_at_100"] - 0.005)
            weights, train_metrics = tune_weights(
                contributions, cube.relevant, train_mask, fold_target
            )
            validation_scores = weighted_scores(contributions, weights)
            validation_metrics = ranking_metrics_from_scores(
                validation_scores, cube.relevant, validation_mask
            )
            fold_rows.append(
                {
                    "fold": int(fold),
                    "method": method,
                    "normalization": normalization or "rank",
                    "weights": json.dumps(_weight_dict(weights), sort_keys=True),
                    "minimum_train_recall_at_100": fold_target,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                }
            )
            method_candidates[(method, normalization)].append(
                rank_feature_rows(
                    cube,
                    validation_scores,
                    top_k=100,
                    query_mask=validation_mask,
                )
            )

    comparison_rows = [
        {
            "method": "rrf",
            "normalization": "rank",
            "variant": "equal_weights_baseline",
            **equal_metrics,
        }
    ]
    meta_candidates_by_method: dict[tuple[str, str | None], pd.DataFrame] = {}
    for method, normalization in methods:
        candidates = pd.concat(
            method_candidates[(method, normalization)], ignore_index=True
        ).sort_values(["query_id", "candidate_rank"], ignore_index=True)
        meta_candidates_by_method[(method, normalization)] = candidates
        comparison_rows.append(
            {
                "method": method,
                "normalization": normalization or "rank",
                "variant": "nested_oof_tuned",
                **metrics_from_candidates(candidates, cube),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    eligible = comparison[
        (comparison["variant"] == "nested_oof_tuned")
        & (comparison["recall_at_100"] >= target_recall)
    ]
    if eligible.empty:
        eligible = comparison[comparison["variant"] == "nested_oof_tuned"]
    best_row = eligible.sort_values(
        ["map_at_10", "recall_at_100"], ascending=False, kind="stable"
    ).iloc[0]
    best_method = str(best_row["method"])
    best_normalization = (
        None if best_row["normalization"] == "rank" else str(best_row["normalization"])
    )
    best_contributions = contribution_tensor(
        cube,
        best_method,
        rrf_k=config.fusion.rrf_k,
        normalization=best_normalization,
    )
    final_weights_array, _ = tune_weights(
        best_contributions,
        cube.relevant,
        np.ones(cube.query_count, dtype=bool),
        target_recall,
    )
    production_scores = weighted_scores(best_contributions, final_weights_array)
    production_candidates = rank_feature_rows(cube, production_scores, top_k=100)
    best_meta_candidates = meta_candidates_by_method[(best_method, best_normalization)]
    ablations = run_ablations(cube, best_contributions, final_weights_array)
    return FusionExperimentResult(
        comparison=comparison,
        meta_folds=pd.DataFrame(fold_rows),
        ablations=ablations,
        best_method=best_method,
        best_normalization=best_normalization,
        final_weights=_weight_dict(final_weights_array),
        meta_candidates=best_meta_candidates,
        production_oof_candidates=production_candidates,
    )


def run_ablations(
    cube: FeatureCube,
    contributions: np.ndarray,
    best_weights: np.ndarray,
) -> pd.DataFrame:
    variants: list[tuple[str, np.ndarray]] = []
    for channel in CHANNELS:
        weights = best_weights.copy()
        weights[CHANNELS.index(channel)] = 0.0
        variants.append((f"without_{channel}", weights))
    query_only = np.zeros(len(CHANNELS), dtype=np.float32)
    query_only[CHANNELS.index("query_knn")] = 1.0
    variants.append(("query_knn_only", query_only))
    lexical = best_weights.copy()
    lexical[CHANNELS.index("dense")] = 0.0
    lexical[CHANNELS.index("query_knn")] = 0.0
    variants.append(("lexical_only", lexical))
    rows = []
    for name, weights in variants:
        rows.append(
            {
                "ablation": name,
                "weights": json.dumps(_weight_dict(weights), sort_keys=True),
                **ranking_metrics_from_scores(
                    weighted_scores(contributions, weights), cube.relevant
                ),
            }
        )
    return pd.DataFrame(rows)


def rank_feature_rows(
    cube: FeatureCube,
    scores: np.ndarray,
    top_k: int,
    query_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    selected_queries = (
        np.flatnonzero(query_mask)
        if query_mask is not None
        else np.arange(cube.query_count)
    )
    order = np.argsort(-scores[selected_queries], axis=1, kind="stable")[:, :top_k]
    flat_indices = (
        selected_queries[:, None] * cube.article_count + order
    ).reshape(-1)
    result = cube.frame.iloc[flat_indices].copy().reset_index(drop=True)
    result.insert(2, "candidate_rank", np.tile(np.arange(1, top_k + 1), len(selected_queries)))
    result.insert(3, "fusion_score", np.take_along_axis(scores[selected_queries], order, axis=1).reshape(-1))
    return result


def metrics_from_candidates(candidates: pd.DataFrame, cube: FeatureCube) -> dict[str, float]:
    query_index = {int(query_id): index for index, query_id in enumerate(cube.query_ids)}
    article_index = {int(article_id): index for index, article_id in enumerate(cube.article_ids)}
    scores = np.full((cube.query_count, cube.article_count), -np.inf, dtype=np.float32)
    for row in candidates.itertuples(index=False):
        scores[query_index[int(row.query_id)], article_index[int(row.article_id)]] = -float(
            row.candidate_rank
        )
    return ranking_metrics_from_scores(scores, cube.relevant)


def analyze_misses(
    meta_candidates: pd.DataFrame,
    features: pd.DataFrame,
    calibration: pd.DataFrame,
    articles: pd.DataFrame,
    config: CandidateGenerationConfig,
) -> pd.DataFrame:
    top100 = {
        int(query_id): set(group["article_id"].astype(int))
        for query_id, group in meta_candidates.groupby("query_id", sort=False)
    }
    feature_lookup = features.set_index(["query_id", "article_id"])
    title_lookup = dict(zip(articles["article_id"].astype(int), articles["title"].astype(str)))
    folds = grouped_multilabel_folds(
        calibration, config.evaluation.folds, config.evaluation.seed
    )
    fold_by_query: dict[int, int] = {}
    train_frequency: dict[int, Counter[int]] = {}
    for fold, (train_indices, validation_indices) in enumerate(folds):
        frequency: Counter[int] = Counter()
        for value in calibration.iloc[train_indices]["ground_truth"].astype(str):
            frequency.update(int(item) for item in value.split())
        train_frequency[fold] = frequency
        for query_id in calibration.iloc[validation_indices]["query_id"].astype(int):
            fold_by_query[query_id] = fold
    global_frequency: Counter[int] = Counter()
    for value in calibration["ground_truth"].astype(str):
        global_frequency.update(int(item) for item in value.split())
    rows: list[dict[str, object]] = []
    for query in calibration.itertuples(index=False):
        query_id = int(query.query_id)
        fold = fold_by_query[query_id]
        for article_id in (int(item) for item in str(query.ground_truth).split()):
            if article_id in top100.get(query_id, set()):
                continue
            feature = feature_lookup.loc[(query_id, article_id)]
            channel_ranks = {
                name: _nullable_int(feature[f"{name}_rank"]) for name in CHANNELS
            }
            flags = []
            if all(rank is None for rank in channel_ranks.values()):
                flags.append("not_retrieved_by_any_channel")
            else:
                flags.append("retrieved_but_below_top100")
            if train_frequency[fold][article_id] == 0:
                flags.append("unseen_label_in_fold_train")
            if global_frequency[article_id] <= 2:
                flags.append("rare_label")
            row: dict[str, object] = {
                "query_id": query_id,
                "query_text": str(query.query_text),
                "fold": fold,
                "missing_article_id": article_id,
                "missing_article_title": title_lookup.get(article_id, ""),
                "reason": ";".join(flags),
                "global_label_frequency": global_frequency[article_id],
                "train_fold_label_frequency": train_frequency[fold][article_id],
            }
            for name in CHANNELS:
                row[f"{name}_rank"] = channel_ranks[name]
                row[f"{name}_score"] = float(feature[f"{name}_score"])
            rows.append(row)
    return pd.DataFrame(rows)


def add_reranker_text(
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    articles: pd.DataFrame,
    chunks: pd.DataFrame,
) -> pd.DataFrame:
    query_text = dict(zip(queries["query_id"].astype(int), queries["query_text"].astype(str)))
    article_title = dict(zip(articles["article_id"].astype(int), articles["title"].astype(str)))
    chunk_text = dict(zip(chunks["chunk_id"].astype(str), chunks["clean_text"].astype(str)))
    result = candidates.copy()
    legacy_bm25 = "best_bm25_chunk_chunk_id"
    canonical_bm25 = "best_bm25_chunk_id"
    if legacy_bm25 in result and canonical_bm25 not in result:
        result = result.rename(columns={legacy_bm25: canonical_bm25})
    result.insert(1, "query_text", result["query_id"].astype(int).map(query_text))
    result.insert(4, "article_title", result["article_id"].astype(int).map(article_title))
    for channel in ("bm25_chunk", "dense"):
        chunk_id_column = (
            canonical_bm25 if channel == "bm25_chunk" else "best_dense_chunk_id"
        )
        result[f"{channel}_text"] = result[chunk_id_column].map(
            lambda value: chunk_text.get(str(value), "") if pd.notna(value) else ""
        )
    return result


def write_experiment_report(
    result: FusionExperimentResult,
    misses: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(output_dir / "fusion_comparison.csv", index=False)
    result.meta_folds.to_csv(output_dir / "fusion_meta_folds.csv", index=False)
    result.ablations.to_csv(output_dir / "ablations.csv", index=False)
    misses.to_csv(output_dir / "misses.csv", index=False, encoding="utf-8-sig")
    write_json(
        {
            "best_method": result.best_method,
            "score_normalization": result.best_normalization,
            "weights": result.final_weights,
            "missed_relevant_pairs": len(misses),
            "missed_queries": int(misses["query_id"].nunique()) if not misses.empty else 0,
        },
        output_dir / "best_config.json",
    )
    lines = [
        "# Fusion experiment",
        "",
        f"- Method: `{result.best_method}`",
        f"- Normalization: `{result.best_normalization or 'rank'}`",
        f"- Weights: `{json.dumps(result.final_weights, ensure_ascii=False, sort_keys=True)}`",
        f"- Missed relevant pairs at 100: `{len(misses)}`",
        "",
        "## Comparison",
        "",
        _markdown_table(result.comparison),
        "",
        "## Ablations",
        "",
        _markdown_table(result.ablations.drop(columns=["weights"])),
    ]
    if not misses.empty:
        miss_columns = [
            "query_id",
            "query_text",
            "missing_article_id",
            "missing_article_title",
            "reason",
            "bm25_title_rank",
            "bm25_chunk_rank",
            "char_tfidf_rank",
            "dense_rank",
            "query_knn_rank",
        ]
        lines.extend(
            [
                "",
                "## Missed relevant articles",
                "",
                _markdown_table(misses[miss_columns].fillna("")),
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _weight_dict(weights: Sequence[float]) -> dict[str, float]:
    return {
        name: round(float(value), 6) for name, value in zip(CHANNELS, weights)
    }


def _nullable_int(value: object) -> int | None:
    return None if pd.isna(value) else int(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
