from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from retrieval.io import read_dataframe


SCORE_COLUMNS = {
    "query_id",
    "article_id",
    "candidate_rank",
    "fusion_score",
    "cross_encoder_score",
}


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    candidate_top_k: int
    rerank_depth: int
    alpha: float


def load_manifest(root: Path) -> tuple[dict[str, object], InferenceConfig]:
    manifest_path = root / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipeline = manifest["pipeline"]
    config = InferenceConfig(
        candidate_top_k=int(pipeline["candidate_top_k"]),
        rerank_depth=int(pipeline["rerank_depth"]),
        alpha=float(pipeline["alpha"]),
    )
    if not 10 <= config.rerank_depth <= config.candidate_top_k:
        raise ValueError("manifest rerank_depth must be in [10, candidate_top_k]")
    if not 0.0 <= config.alpha <= 1.0:
        raise ValueError("manifest alpha must be in [0, 1]")
    return manifest, config


def load_bundled_scores(root: Path, manifest: dict[str, object]) -> pd.DataFrame:
    score_path = root / "artifacts" / "production_scores.f"
    files = manifest["files"]
    expected = str(files["production_scores.f"]["sha256"])
    actual = sha256_file(score_path)
    if actual != expected:
        raise ValueError("production_scores.f failed the SHA-256 integrity check")
    return read_dataframe(score_path)


def build_answer(
    scores: pd.DataFrame,
    test_queries: pd.DataFrame,
    articles: pd.DataFrame,
    config: InferenceConfig,
) -> pd.DataFrame:
    validate_scores(scores, test_queries, articles, config)
    rows: list[dict[str, object]] = []
    grouped = {int(key): value for key, value in scores.groupby("query_id", sort=False)}
    for query_id in test_queries["query_id"].astype(int):
        group = grouped[query_id].sort_values("candidate_rank", kind="stable").copy()
        active = group[group["candidate_rank"] <= config.rerank_depth].copy()
        tail = group[group["candidate_rank"] > config.rerank_depth].copy()
        active["_ce"] = minmax(active["cross_encoder_score"])
        active["_fusion"] = minmax(active["fusion_score"])
        active["final_score"] = (
            config.alpha * active["_ce"]
            + (1.0 - config.alpha) * active["_fusion"]
        )
        active = active.sort_values(
            ["final_score", "candidate_rank", "article_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        ranked = pd.concat([active, tail], ignore_index=True)
        answer = " ".join(str(value) for value in ranked["article_id"].head(10))
        rows.append({"query_id": query_id, "answer": answer})
    result = pd.DataFrame(rows, columns=["query_id", "answer"])
    validate_answer(result, test_queries, articles)
    return result


def validate_scores(
    scores: pd.DataFrame,
    test_queries: pd.DataFrame,
    articles: pd.DataFrame,
    config: InferenceConfig,
) -> None:
    missing = SCORE_COLUMNS - set(scores.columns)
    if missing:
        raise ValueError(f"score artifact is missing columns: {sorted(missing)}")
    if scores.duplicated(["query_id", "article_id"]).any():
        raise ValueError("score artifact contains duplicate query/article pairs")
    expected_queries = set(test_queries["query_id"].astype(int))
    actual_queries = set(scores["query_id"].astype(int))
    if actual_queries != expected_queries:
        raise ValueError("score artifact does not cover exactly the test queries")
    valid_articles = set(articles["article_id"].astype(int))
    if not set(scores["article_id"].astype(int)) <= valid_articles:
        raise ValueError("score artifact contains unknown article_id values")
    counts = scores.groupby("query_id").size()
    if not (counts == config.candidate_top_k).all():
        raise ValueError(
            f"each query must have exactly {config.candidate_top_k} candidates"
        )
    expected_ranks = list(range(1, config.candidate_top_k + 1))
    for _, ranks in scores.groupby("query_id", sort=False)["candidate_rank"]:
        if sorted(int(value) for value in ranks) != expected_ranks:
            raise ValueError("candidate ranks must cover 1..candidate_top_k")
    fusion = pd.to_numeric(scores["fusion_score"], errors="coerce")
    if not np.isfinite(fusion).all():
        raise ValueError("score artifact contains non-finite fusion scores")
    active = scores["candidate_rank"] <= config.rerank_depth
    cross_encoder = pd.to_numeric(scores["cross_encoder_score"], errors="coerce")
    if not np.isfinite(cross_encoder[active]).all():
        raise ValueError("top rerank_depth candidates require finite GTE scores")
    if cross_encoder[~active].notna().any():
        raise ValueError("candidates below rerank_depth must not contain GTE scores")


def validate_answer(
    answer: pd.DataFrame,
    test_queries: pd.DataFrame,
    articles: pd.DataFrame,
) -> None:
    if answer.columns.tolist() != ["query_id", "answer"]:
        raise ValueError("answer.csv must contain only query_id and answer")
    if answer["query_id"].duplicated().any():
        raise ValueError("answer.csv contains duplicate query_id values")
    if answer["query_id"].astype(int).tolist() != test_queries["query_id"].astype(int).tolist():
        raise ValueError("answer.csv query order or coverage differs from test.f")
    valid_articles = set(articles["article_id"].astype(int))
    for value in answer["answer"].astype(str):
        article_ids = [int(item) for item in value.split()]
        if len(article_ids) != 10 or len(set(article_ids)) != 10:
            raise ValueError("each answer must contain 10 unique article IDs")
        if not set(article_ids) <= valid_articles:
            raise ValueError("answer.csv contains an unknown article_id")


def minmax(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum == minimum:
        return np.ones(len(array), dtype=np.float64)
    return (array - minimum) / (maximum - minimum)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
