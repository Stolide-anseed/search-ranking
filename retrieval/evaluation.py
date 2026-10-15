from __future__ import annotations

import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from preprocessing import normalize_text
from retrieval.channels import QueryKnnChannel
from retrieval.config import CandidateGenerationConfig
from retrieval.dense import DenseEncoder
from retrieval.hybrid import fuse_channel_results
from retrieval.index import UniversalIndex


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, dict[str, float]]
    fold_metrics: pd.DataFrame
    candidates: pd.DataFrame
    diagnostics: dict[str, object]


def grouped_multilabel_folds(
    calibration: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    normalized = calibration["query_text"].astype(str).map(normalize_text)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(normalized):
        groups[value].append(index)
    group_items = list(groups.values())
    all_labels = sorted(
        {
            int(article_id)
            for value in calibration["ground_truth"].astype(str)
            for article_id in value.split()
        }
    )
    label_to_index = {label: index for index, label in enumerate(all_labels)}
    targets = np.zeros((len(group_items), len(all_labels)), dtype=np.int8)
    for group_index, row_indices in enumerate(group_items):
        for row_index in row_indices:
            for article_id in str(calibration.iloc[row_index]["ground_truth"]).split():
                targets[group_index, label_to_index[int(article_id)]] = 1

    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_groups, validation_groups in splitter.split(
        np.zeros(len(group_items)), targets
    ):
        train_indices = np.asarray(
            [index for group in train_groups for index in group_items[group]],
            dtype=np.int32,
        )
        validation_indices = np.asarray(
            [index for group in validation_groups for index in group_items[group]],
            dtype=np.int32,
        )
        folds.append((np.sort(train_indices), np.sort(validation_indices)))
    return folds


def evaluate_oof(
    index: UniversalIndex,
    encoder: DenseEncoder,
    calibration: pd.DataFrame,
    config: CandidateGenerationConfig,
) -> EvaluationResult:
    folds = grouped_multilabel_folds(
        calibration,
        n_splits=config.evaluation.folds,
        seed=config.evaluation.seed,
    )
    universal_channels = index.channels(encoder)
    experiments = {
        "bm25_chunk": ("bm25_chunk",),
        "char_tfidf": ("char_tfidf",),
        "dense": ("dense",),
        "query_knn": ("query_knn",),
        "lexical_hybrid": ("bm25_chunk", "char_tfidf"),
        "universal_hybrid": (
            "bm25_chunk",
            "char_tfidf",
            "dense",
        ),
        "full_hybrid": (
            "bm25_chunk",
            "char_tfidf",
            "dense",
            "query_knn",
        ),
    }
    by_experiment: dict[str, list[pd.DataFrame]] = defaultdict(list)
    fold_rows: list[dict[str, object]] = []
    full_candidates: list[pd.DataFrame] = []
    channel_timings: dict[str, float] = defaultdict(float)
    tracemalloc.start()
    evaluation_start = time.perf_counter()

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
        channel_results: dict[str, pd.DataFrame] = {}
        for name, channel in channels.items():
            started = time.perf_counter()
            # Production linear fusion normalizes over the complete article set.
            channel_results[name] = channel.search(validation, len(index.articles))
            channel_timings[name] += time.perf_counter() - started

        for experiment_name, channel_names in experiments.items():
            started = time.perf_counter()
            selected = {name: channel_results[name] for name in channel_names}
            selected_weights = {
                name: config.fusion.weights[name] for name in channel_names
            }
            candidates = fuse_channel_results(
                queries=validation,
                article_ids=index.articles["article_id"].astype(int).tolist(),
                channel_results=selected,
                weights=selected_weights,
                rrf_k=config.fusion.rrf_k,
                top_k=config.retrieval.candidate_top_k,
                method=config.fusion.method,
                score_normalization=config.fusion.score_normalization,
            )
            candidates["fold"] = fold_index
            metrics = ranking_metrics(candidates, validation)
            fold_rows.append(
                {
                    "fold": fold_index,
                    "experiment": experiment_name,
                    "queries": len(validation),
                    "duration_seconds": time.perf_counter() - started,
                    **metrics,
                }
            )
            by_experiment[experiment_name].append(candidates)
            if experiment_name == "full_hybrid":
                full_candidates.append(candidates)

    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    combined_metrics = {
        name: ranking_metrics(
            pd.concat(candidate_frames, ignore_index=True), calibration
        )
        for name, candidate_frames in by_experiment.items()
    }
    best_single_recall = max(
        combined_metrics[name]["recall_at_100"]
        for name in ("bm25_chunk", "char_tfidf", "dense", "query_knn")
    )
    full_recall = combined_metrics["full_hybrid"]["recall_at_100"]
    diagnostics = {
        "duration_seconds": round(time.perf_counter() - evaluation_start, 4),
        "peak_python_memory_bytes": int(peak_memory),
        "channel_search_seconds": {
            name: round(duration, 4) for name, duration in channel_timings.items()
        },
        "full_hybrid_meets_recall_criterion": full_recall >= best_single_recall,
        "best_single_recall_at_100": best_single_recall,
        "full_hybrid_recall_at_100": full_recall,
    }
    try:
        import torch

        diagnostics["peak_cuda_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        )
    except ImportError:
        diagnostics["peak_cuda_memory_bytes"] = 0
    return EvaluationResult(
        metrics=combined_metrics,
        fold_metrics=pd.DataFrame(fold_rows),
        candidates=pd.concat(full_candidates, ignore_index=True),
        diagnostics=diagnostics,
    )


def ranking_metrics(
    candidates: pd.DataFrame,
    queries_with_truth: pd.DataFrame,
) -> dict[str, float]:
    rankings = {
        int(query_id): group.sort_values("candidate_rank")["article_id"].astype(int).tolist()
        for query_id, group in candidates.groupby("query_id", sort=False)
    }
    average_precisions: list[float] = []
    recalls: dict[int, list[float]] = {10: [], 20: [], 50: [], 100: []}
    for row in queries_with_truth.itertuples(index=False):
        relevant = {int(item) for item in str(row.ground_truth).split()}
        ranking = rankings.get(int(row.query_id), [])
        hits = 0
        precision_sum = 0.0
        seen: set[int] = set()
        for rank, article_id in enumerate(ranking[:10], start=1):
            if article_id in relevant and article_id not in seen:
                hits += 1
                precision_sum += hits / rank
            seen.add(article_id)
        average_precisions.append(precision_sum / min(len(relevant), 10))
        for cutoff in recalls:
            recalls[cutoff].append(len(set(ranking[:cutoff]) & relevant) / len(relevant))
    return {
        "map_at_10": round(float(np.mean(average_precisions)), 6),
        **{
            f"recall_at_{cutoff}": round(float(np.mean(values)), 6)
            for cutoff, values in recalls.items()
        },
    }
