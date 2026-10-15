from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from retrieval.channels import CandidateChannel


BEST_CHUNK_CHANNELS = {"bm25_chunk", "dense"}


def fuse_channel_results(
    queries: pd.DataFrame,
    article_ids: Sequence[int],
    channel_results: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    rrf_k: int = 60,
    top_k: int = 100,
    method: str = "rrf",
    score_normalization: str | None = None,
) -> pd.DataFrame:
    if method not in {"rrf", "linear"}:
        raise ValueError(f"unknown fusion method: {method}")
    if method == "linear" and score_normalization not in {"max", "minmax"}:
        raise ValueError("linear fusion requires max or minmax score normalization")
    query_ids = queries["query_id"].astype(int).tolist()
    merged: pd.DataFrame | None = None
    for channel_name, result in channel_results.items():
        best_chunk_column = (
            "best_bm25_chunk_id"
            if channel_name == "bm25_chunk"
            else f"best_{channel_name}_chunk_id"
        )
        selected = result[["query_id", "article_id", "score", "rank", "best_chunk_id"]].copy()
        selected = selected.rename(
            columns={
                "score": f"{channel_name}_score",
                "rank": f"{channel_name}_rank",
                "best_chunk_id": best_chunk_column,
            }
        )
        if channel_name not in BEST_CHUNK_CHANNELS:
            selected = selected.drop(columns=[best_chunk_column])
        merged = (
            selected
            if merged is None
            else merged.merge(selected, on=["query_id", "article_id"], how="outer")
        )

    if merged is None:
        merged = pd.DataFrame(columns=["query_id", "article_id"])
    valid_article_ids = tuple(sorted(int(article_id) for article_id in article_ids))
    output: list[pd.DataFrame] = []
    for query_id in query_ids:
        query_rows = merged[merged["query_id"] == query_id].copy()
        present = set(query_rows["article_id"].dropna().astype(int))
        if len(query_rows) < top_k:
            fallback = [
                article_id for article_id in valid_article_ids if article_id not in present
            ][: top_k - len(query_rows)]
            if fallback:
                filler = pd.DataFrame(
                    {"query_id": query_id, "article_id": fallback}
                )
                query_rows = pd.concat([query_rows, filler], ignore_index=True, sort=False)

        query_rows["fusion_score"] = 0.0
        for channel_name, weight in weights.items():
            rank_column = f"{channel_name}_rank"
            if rank_column not in query_rows:
                continue
            ranks = pd.to_numeric(query_rows[rank_column], errors="coerce")
            if method == "rrf":
                contribution = np.where(
                    ranks.notna(), float(weight) / (rrf_k + ranks.fillna(0)), 0.0
                )
            else:
                raw_scores = pd.to_numeric(
                    query_rows[f"{channel_name}_score"], errors="coerce"
                ).fillna(0.0)
                contribution = float(weight) * _normalize_query_scores(
                    raw_scores.to_numpy(dtype=np.float64),
                    ranks.notna().to_numpy(),
                    str(score_normalization),
                )
            query_rows["fusion_score"] += contribution
        # Kept as a compatibility alias for the original candidate schema.
        query_rows["rrf_score"] = query_rows["fusion_score"]
        query_rows = query_rows.sort_values(
            ["fusion_score", "article_id"], ascending=[False, True], kind="stable"
        ).head(top_k)
        query_rows.insert(2, "candidate_rank", range(1, len(query_rows) + 1))
        output.append(query_rows)

    if not output:
        columns = [
            "query_id",
            "article_id",
            "candidate_rank",
            "fusion_score",
            "rrf_score",
        ]
        for channel_name in channel_results:
            columns.extend(
                [f"{channel_name}_score", f"{channel_name}_rank"]
            )
            if channel_name in BEST_CHUNK_CHANNELS:
                columns.append(
                    "best_bm25_chunk_id"
                    if channel_name == "bm25_chunk"
                    else f"best_{channel_name}_chunk_id"
                )
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))

    candidates = pd.concat(output, ignore_index=True, sort=False)
    score_columns = [column for column in candidates if column.endswith("_score")]
    rank_columns = [column for column in candidates if column.endswith("_rank")]
    candidates[score_columns] = candidates[score_columns].fillna(0.0)
    for column in rank_columns:
        candidates[column] = candidates[column].astype("Int64")
    ordered = [
        "query_id",
        "article_id",
        "candidate_rank",
        "fusion_score",
        "rrf_score",
    ]
    remainder = [column for column in candidates if column not in ordered]
    return candidates[ordered + sorted(remainder)]


@dataclass(slots=True)
class HybridCandidateGenerator:
    article_ids: Sequence[int]
    channels: Mapping[str, CandidateChannel]
    weights: Mapping[str, float]
    rrf_k: int = 60
    channel_top_k: int | Mapping[str, int] = 100
    method: str = "rrf"
    score_normalization: str | None = None

    def generate(self, queries: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
        _validate_queries(queries)
        results = {
            name: channel.search(queries, self._top_k_for(name))
            for name, channel in self.channels.items()
        }
        return fuse_channel_results(
            queries=queries,
            article_ids=self.article_ids,
            channel_results=results,
            weights=self.weights,
            rrf_k=self.rrf_k,
            top_k=top_k,
            method=self.method,
            score_normalization=self.score_normalization,
        )

    def channel_results(self, queries: pd.DataFrame) -> dict[str, pd.DataFrame]:
        _validate_queries(queries)
        return {
            name: channel.search(queries, self._top_k_for(name))
            for name, channel in self.channels.items()
        }

    def _top_k_for(self, channel_name: str) -> int:
        if isinstance(self.channel_top_k, Mapping):
            return int(self.channel_top_k[channel_name])
        return int(self.channel_top_k)


def _validate_queries(queries: pd.DataFrame) -> None:
    missing = {"query_id", "query_text"} - set(queries.columns)
    if missing:
        raise ValueError(f"missing query columns: {sorted(missing)}")
    if queries["query_id"].duplicated().any():
        raise ValueError("query_id values must be unique")


def _normalize_query_scores(
    scores: np.ndarray,
    present: np.ndarray,
    method: str,
) -> np.ndarray:
    result = np.zeros_like(scores, dtype=np.float64)
    values = scores[present]
    if not len(values):
        return result
    if method == "max":
        maximum = float(values.max())
        if maximum > 0:
            result[present] = values / maximum
        return result
    if method == "minmax":
        minimum = float(values.min())
        maximum = float(values.max())
        if maximum > minimum:
            result[present] = (values - minimum) / (maximum - minimum)
        else:
            result[present] = 1.0
        return result
    raise ValueError(f"unknown score normalization: {method}")
