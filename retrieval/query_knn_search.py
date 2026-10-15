from __future__ import annotations

import itertools
import json
import platform
import sys
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval.fusion_experiments import (
    CHANNELS,
    add_reranker_text,
    contribution_tensor,
    make_feature_cube,
    rank_feature_rows,
    ranking_metrics_from_scores,
)
from retrieval.io import read_dataframe, write_dataframe, write_json


@dataclass(frozen=True, slots=True)
class QueryKnnParameters:
    neighbors: int
    frequency_penalty: float
    word_weight: float
    voting: str


@dataclass(frozen=True, slots=True)
class QueryKnnSearchConfig:
    features: Path
    calibration: Path
    baseline_folds: Path
    articles: Path
    chunks: Path
    output_dir: Path
    neighbors: tuple[int, ...]
    frequency_penalties: tuple[float, ...]
    word_weights: tuple[float, ...]
    voting: tuple[str, ...]
    method: str
    normalization: str
    minimum_recall_at_100: float
    folds: int
    top_k: int
    seed: int


@dataclass(slots=True)
class PreparedSimilarities:
    word: np.ndarray
    char: np.ndarray
    labels: list[tuple[int, ...]]
    label_frequency: Counter[int]


@dataclass(slots=True)
class QueryKnnSearchResult:
    metrics: dict[str, dict[str, float]]
    fold_metrics: pd.DataFrame
    inner_grid: pd.DataFrame
    oof_features: pd.DataFrame
    oof_candidates: pd.DataFrame


def load_query_knn_search_config(path: Path) -> QueryKnnSearchConfig:
    with path.open("rb") as source:
        raw = tomllib.load(source)
    base = path.parent.parent.resolve()

    def project(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else base / candidate

    paths = raw["paths"]
    search = raw["search"]
    fusion = raw["fusion"]
    evaluation = raw["evaluation"]
    config = QueryKnnSearchConfig(
        features=project(paths["features"]),
        calibration=project(paths["calibration"]),
        baseline_folds=project(paths["baseline_folds"]),
        articles=project(paths["articles"]),
        chunks=project(paths["chunks"]),
        output_dir=project(paths["output_dir"]),
        neighbors=tuple(int(value) for value in search["neighbors"]),
        frequency_penalties=tuple(float(value) for value in search["frequency_penalties"]),
        word_weights=tuple(float(value) for value in search["word_weights"]),
        voting=tuple(str(value) for value in search["voting"]),
        method=str(fusion["method"]),
        normalization=str(fusion["normalization"]),
        minimum_recall_at_100=float(fusion["minimum_recall_at_100"]),
        folds=int(evaluation["folds"]),
        top_k=int(evaluation["top_k"]),
        seed=int(evaluation["seed"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: QueryKnnSearchConfig) -> None:
    if config.method != "linear" or config.normalization != "max":
        raise ValueError("query-kNN search currently requires linear/max fusion")
    if not config.neighbors or any(value <= 0 for value in config.neighbors):
        raise ValueError("neighbors must contain positive values")
    if any(value < 0 for value in config.frequency_penalties):
        raise ValueError("frequency penalties must be non-negative")
    if any(not 0 <= value <= 1 for value in config.word_weights):
        raise ValueError("word weights must be in [0, 1]")
    if set(config.voting) - {"similarity_sum", "count", "max"}:
        raise ValueError("unsupported query-kNN voting method")


def prepare_similarities(train: pd.DataFrame, validation: pd.DataFrame) -> PreparedSimilarities:
    train_text = train["query_text"].astype(str).tolist()
    validation_text = validation["query_text"].astype(str).tolist()
    labels = [tuple(int(item) for item in value.split()) for value in train["ground_truth"].astype(str)]
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_train = word_vectorizer.fit_transform(train_text)
    char_train = char_vectorizer.fit_transform(train_text)
    word = (word_vectorizer.transform(validation_text) @ word_train.T).toarray().astype(np.float32)
    char = (char_vectorizer.transform(validation_text) @ char_train.T).toarray().astype(np.float32)
    return PreparedSimilarities(
        word=word,
        char=char,
        labels=labels,
        label_frequency=Counter(article_id for values in labels for article_id in values),
    )


def score_prepared(
    prepared: PreparedSimilarities,
    article_ids: np.ndarray,
    parameters: QueryKnnParameters,
) -> np.ndarray:
    similarities = (
        parameters.word_weight * prepared.word
        + (1.0 - parameters.word_weight) * prepared.char
    )
    article_index = {int(article_id): index for index, article_id in enumerate(article_ids)}
    result = np.zeros((len(similarities), len(article_ids)), dtype=np.float32)
    for query_index, scores in enumerate(similarities):
        positive = np.flatnonzero(scores > 0)
        neighbors = positive[
            np.lexsort((positive, -scores[positive]))[: parameters.neighbors]
        ]
        article_scores: dict[int, float] = {}
        for neighbor_index in neighbors:
            similarity = float(scores[neighbor_index])
            for article_id in prepared.labels[neighbor_index]:
                if parameters.voting == "count":
                    article_scores[article_id] = article_scores.get(article_id, 0.0) + 1.0
                elif parameters.voting == "max":
                    article_scores[article_id] = max(article_scores.get(article_id, 0.0), similarity)
                else:
                    article_scores[article_id] = article_scores.get(article_id, 0.0) + similarity
        for article_id, score in article_scores.items():
            index = article_index.get(article_id)
            if index is None:
                continue
            frequency = prepared.label_frequency[article_id]
            result[query_index, index] = score / frequency**parameters.frequency_penalty
    return result


def normalize_max(scores: np.ndarray) -> np.ndarray:
    maxima = scores.max(axis=1, keepdims=True)
    return np.divide(scores, maxima, out=np.zeros_like(scores), where=maxima > 0)


def score_to_ranks(scores: np.ndarray) -> np.ndarray:
    ranks = np.zeros(scores.shape, dtype=np.int32)
    for query_index, values in enumerate(scores):
        present = np.flatnonzero(values > 0)
        order = present[np.lexsort((present, -values[present]))]
        ranks[query_index, order] = np.arange(1, len(order) + 1)
    return ranks


def _load_fold_weights(path: Path, method: str, normalization: str) -> dict[int, np.ndarray]:
    rows = pd.read_csv(path)
    rows = rows[(rows["method"] == method) & (rows["normalization"] == normalization)]
    result = {
        int(row.fold): np.asarray(
            [float(json.loads(row.weights)[name]) for name in CHANNELS], dtype=np.float32
        )
        for row in rows.itertuples(index=False)
    }
    if not result:
        raise ValueError("no matching fold weights found")
    return result


def _fusion_with_knn(
    contributions: np.ndarray,
    query_indices: np.ndarray,
    fold_weights: np.ndarray,
    knn_scores: np.ndarray,
) -> np.ndarray:
    lexical_dense = np.tensordot(
        fold_weights[:-1], contributions[:-1, query_indices], axes=(0, 0)
    )
    return lexical_dense + fold_weights[-1] * normalize_max(knn_scores)


def _parameter_key(
    metrics: Mapping[str, float], parameters: QueryKnnParameters, minimum_recall: float
) -> tuple[float, ...]:
    return (
        float(metrics["recall_at_100"] >= minimum_recall),
        float(metrics["map_at_10"]),
        float(metrics["recall_at_10"]),
        float(metrics["recall_at_50"]),
        -abs(parameters.neighbors - 50),
        -abs(parameters.frequency_penalty - 0.5),
        -abs(parameters.word_weight - 0.5),
        float(parameters.voting == "similarity_sum"),
    )


def evaluate_query_knn_search(
    features: pd.DataFrame,
    calibration: pd.DataFrame,
    fold_weights: Mapping[int, np.ndarray],
    config: QueryKnnSearchConfig,
) -> QueryKnnSearchResult:
    cube = make_feature_cube(features)
    if set(np.unique(cube.folds)) != set(range(config.folds)):
        raise ValueError("feature folds do not match query-kNN search configuration")
    contributions = contribution_tensor(cube, config.method, normalization=config.normalization)
    query_lookup = calibration.set_index(calibration["query_id"].astype(int), drop=False)
    ordered_queries = query_lookup.loc[cube.query_ids].reset_index(drop=True)
    baseline_scores = np.zeros((cube.query_count, cube.article_count), dtype=np.float32)
    tuned_scores = np.zeros_like(baseline_scores)
    tuned_knn_raw = np.zeros_like(baseline_scores)
    fold_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    parameter_grid = [
        QueryKnnParameters(*values)
        for values in itertools.product(
            config.neighbors,
            config.frequency_penalties,
            config.word_weights,
            config.voting,
        )
    ]

    for outer_fold in range(config.folds):
        outer_indices = np.flatnonzero(cube.folds == outer_fold)
        baseline_scores[outer_indices] = np.tensordot(
            fold_weights[outer_fold], contributions[:, outer_indices], axes=(0, 0)
        )
        inner_results: dict[QueryKnnParameters, list[dict[str, float]]] = {
            parameters: [] for parameters in parameter_grid
        }
        for inner_fold in range(config.folds):
            if inner_fold == outer_fold:
                continue
            inner_indices = np.flatnonzero(cube.folds == inner_fold)
            train_indices = np.flatnonzero(
                ~np.isin(cube.folds, [outer_fold, inner_fold])
            )
            prepared_inner = prepare_similarities(
                ordered_queries.iloc[train_indices], ordered_queries.iloc[inner_indices]
            )
            for parameters in parameter_grid:
                raw_scores = score_prepared(prepared_inner, cube.article_ids, parameters)
                scores = _fusion_with_knn(
                    contributions, inner_indices, fold_weights[inner_fold], raw_scores
                )
                metrics = ranking_metrics_from_scores(scores, cube.relevant[inner_indices])
                inner_results[parameters].append(metrics)
                grid_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        **asdict(parameters),
                        **metrics,
                    }
                )
        best_parameters: QueryKnnParameters | None = None
        best_key: tuple[float, ...] | None = None
        best_inner_metrics: dict[str, float] | None = None
        for parameters, rows in inner_results.items():
            metrics = {
                name: round(float(np.mean([row[name] for row in rows])), 6)
                for name in rows[0]
            }
            key = _parameter_key(metrics, parameters, config.minimum_recall_at_100)
            if best_key is None or key > best_key:
                best_key = key
                best_parameters = parameters
                best_inner_metrics = metrics
        assert best_parameters is not None and best_inner_metrics is not None

        refit_indices = np.flatnonzero(cube.folds != outer_fold)
        prepared_outer = prepare_similarities(
            ordered_queries.iloc[refit_indices], ordered_queries.iloc[outer_indices]
        )
        outer_knn = score_prepared(prepared_outer, cube.article_ids, best_parameters)
        outer_scores = _fusion_with_knn(
            contributions, outer_indices, fold_weights[outer_fold], outer_knn
        )
        tuned_knn_raw[outer_indices] = outer_knn
        tuned_scores[outer_indices] = outer_scores
        baseline_metrics = ranking_metrics_from_scores(
            baseline_scores, cube.relevant, cube.folds == outer_fold
        )
        outer_metrics = ranking_metrics_from_scores(
            tuned_scores, cube.relevant, cube.folds == outer_fold
        )
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": "all_non_outer",
                **asdict(best_parameters),
                **{f"inner_{key}": value for key, value in best_inner_metrics.items()},
                **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                **outer_metrics,
                "map_at_10_delta": round(
                    outer_metrics["map_at_10"] - baseline_metrics["map_at_10"], 6
                ),
            }
        )

    updated = cube.frame.copy()
    updated["query_knn_score"] = tuned_knn_raw.reshape(-1).astype(np.float32)
    rank_values = score_to_ranks(tuned_knn_raw).reshape(-1)
    updated["query_knn_rank"] = pd.array(
        np.where(rank_values > 0, rank_values, None), dtype="Int32"
    )
    updated_cube = make_feature_cube(updated)
    candidates = rank_feature_rows(updated_cube, tuned_scores, config.top_k)
    metrics = {
        "fixed_bge_fusion": ranking_metrics_from_scores(baseline_scores, cube.relevant),
        "tuned_query_knn_fusion": ranking_metrics_from_scores(tuned_scores, cube.relevant),
        "original_query_knn_only": ranking_metrics_from_scores(
            contributions[CHANNELS.index("query_knn")], cube.relevant
        ),
        "tuned_query_knn_only": ranking_metrics_from_scores(
            normalize_max(tuned_knn_raw), cube.relevant
        ),
    }
    return QueryKnnSearchResult(
        metrics=metrics,
        fold_metrics=pd.DataFrame(fold_rows),
        inner_grid=pd.DataFrame(grid_rows),
        oof_features=updated,
        oof_candidates=candidates,
    )


def run_query_knn_search(config: QueryKnnSearchConfig) -> QueryKnnSearchResult:
    features = read_dataframe(config.features)
    calibration = read_dataframe(config.calibration)
    fold_weights = _load_fold_weights(
        config.baseline_folds, config.method, config.normalization
    )
    result = evaluate_query_knn_search(features, calibration, fold_weights, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = result.metrics["fixed_bge_fusion"]
    tuned = result.metrics["tuned_query_knn_fusion"]
    improved_folds = int((result.fold_metrics["map_at_10_delta"] > 0).sum())
    acceptance = {
        "accepted": bool(
            tuned["map_at_10"] > fixed["map_at_10"]
            and improved_folds >= 4
            and tuned["recall_at_100"] >= config.minimum_recall_at_100
        ),
        "improved_folds": improved_folds,
        "required_improved_folds": 4,
        "map_at_10_delta": round(tuned["map_at_10"] - fixed["map_at_10"], 6),
    }
    write_json(
        {"metrics": result.metrics, "acceptance": acceptance},
        config.output_dir / "metrics.json",
    )
    result.fold_metrics.to_csv(config.output_dir / "fold_metrics.csv", index=False)
    result.inner_grid.to_csv(config.output_dir / "inner_grid.csv", index=False)
    write_dataframe(result.oof_features, config.output_dir / "oof_features.f")
    write_dataframe(result.oof_candidates, config.output_dir / "oof_candidates.f")
    top50 = result.oof_candidates[result.oof_candidates["candidate_rank"] <= 50].copy()
    top50 = add_reranker_text(
        top50,
        calibration,
        read_dataframe(config.articles),
        read_dataframe(config.chunks),
    )
    write_dataframe(top50, config.output_dir / "reranker_candidates_oof.f")
    serialized = asdict(config)
    for key in ("features", "calibration", "baseline_folds", "articles", "chunks", "output_dir"):
        serialized[key] = str(serialized[key])
    write_json(serialized, config.output_dir / "selected_config.json")
    write_json(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        config.output_dir / "manifest.json",
    )
    return result
