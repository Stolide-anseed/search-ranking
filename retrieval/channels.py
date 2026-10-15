from __future__ import annotations

from collections import Counter
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import normalize_text
from retrieval.bm25 import BM25Index
from retrieval.dense import DenseEncoder


RESULT_COLUMNS = [
    "query_id",
    "article_id",
    "score",
    "rank",
    "best_chunk_id",
]


class CandidateChannel(Protocol):
    name: str

    def search(self, queries: pd.DataFrame, top_k: int) -> pd.DataFrame: ...


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


class BM25ChunkChannel:
    name = "bm25_chunk"

    def __init__(self, k1: float = 1.5, b: float = 0.75, chunk_top_k: int = 300) -> None:
        self.index = BM25Index(k1=k1, b=b)
        self.chunk_top_k = chunk_top_k

    def fit(
        self, articles: pd.DataFrame, chunks: pd.DataFrame
    ) -> "BM25ChunkChannel":
        titles = dict(zip(articles["article_id"].astype(int), articles["title"].astype(str)))
        self.chunk_ids = chunks["chunk_id"].astype(str).to_numpy()
        self.article_ids = chunks["article_id"].astype(int).to_numpy()
        documents = []
        for chunk in chunks.itertuples(index=False):
            path = " ".join(str(item) for item in chunk.section_path)
            documents.append(
                normalize_text(f"{titles[int(chunk.article_id)]} {path} {chunk.clean_text}")
            )
        self.index.fit(documents)
        return self

    def search(self, queries: pd.DataFrame, top_k: int) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for query in queries.itertuples(index=False):
            indices, scores = self.index.search(str(query.query_text), self.chunk_top_k)
            best: dict[int, tuple[float, str]] = {}
            for index, score in zip(indices, scores):
                article_id = int(self.article_ids[index])
                current = best.get(article_id)
                candidate = (float(score), str(self.chunk_ids[index]))
                if current is None or candidate[0] > current[0]:
                    best[article_id] = candidate
            ordered = sorted(best.items(), key=lambda item: (-item[1][0], item[0]))[:top_k]
            for rank, (article_id, (score, chunk_id)) in enumerate(ordered, start=1):
                rows.append(
                    {
                        "query_id": int(query.query_id),
                        "article_id": article_id,
                        "score": score,
                        "rank": rank,
                        "best_chunk_id": chunk_id,
                    }
                )
        return pd.DataFrame(rows, columns=RESULT_COLUMNS) if rows else _empty_results()


class CharTfidfChannel:
    name = "char_tfidf"

    def __init__(self, min_df: int = 2) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=min_df,
            sublinear_tf=True,
            dtype=np.float32,
        )

    def fit(self, articles: pd.DataFrame) -> "CharTfidfChannel":
        self.article_ids = articles["article_id"].astype(int).to_numpy()
        documents = (
            articles["title"].astype(str) + " " + articles["normalized_text"].astype(str)
        ).tolist()
        self.matrix = self.vectorizer.fit_transform(documents)
        return self

    def search(self, queries: pd.DataFrame, top_k: int) -> pd.DataFrame:
        query_matrix = self.vectorizer.transform(queries["query_text"].astype(str))
        similarities = query_matrix @ self.matrix.T
        rows: list[dict[str, object]] = []
        for row_index, query_id in enumerate(queries["query_id"].astype(int)):
            dense_scores = similarities.getrow(row_index).toarray().ravel()
            positive = np.flatnonzero(dense_scores > 0)
            order = positive[np.lexsort((positive, -dense_scores[positive]))[:top_k]]
            for rank, article_index in enumerate(order, start=1):
                rows.append(
                    {
                        "query_id": query_id,
                        "article_id": int(self.article_ids[article_index]),
                        "score": float(dense_scores[article_index]),
                        "rank": rank,
                        "best_chunk_id": None,
                    }
                )
        return pd.DataFrame(rows, columns=RESULT_COLUMNS) if rows else _empty_results()


class DenseChunkChannel:
    name = "dense"

    def __init__(self, encoder: DenseEncoder, chunk_top_k: int = 300) -> None:
        self.encoder = encoder
        self.chunk_top_k = chunk_top_k

    def fit(
        self,
        chunks: pd.DataFrame,
        embeddings: np.ndarray | None = None,
    ) -> "DenseChunkChannel":
        self.chunk_ids = chunks["chunk_id"].astype(str).to_numpy()
        self.article_ids = chunks["article_id"].astype(int).to_numpy()
        self.embeddings = (
            self.encoder.encode(chunks["dense_text"].astype(str).tolist())
            if embeddings is None
            else np.asarray(embeddings, dtype=np.float32)
        )
        if len(self.embeddings) != len(chunks):
            raise ValueError("dense embedding count does not match chunk count")
        return self

    def search(self, queries: pd.DataFrame, top_k: int) -> pd.DataFrame:
        query_embeddings = self.encoder.encode_queries(queries["query_text"].astype(str).tolist())
        similarities = query_embeddings @ self.embeddings.T
        rows: list[dict[str, object]] = []
        for row_index, query_id in enumerate(queries["query_id"].astype(int)):
            scores = similarities[row_index]
            chunk_order = np.lexsort((np.arange(len(scores)), -scores))[: self.chunk_top_k]
            best: dict[int, tuple[float, str]] = {}
            for chunk_index in chunk_order:
                article_id = int(self.article_ids[chunk_index])
                candidate = (float(scores[chunk_index]), str(self.chunk_ids[chunk_index]))
                current = best.get(article_id)
                if current is None or candidate[0] > current[0]:
                    best[article_id] = candidate
            ordered = sorted(best.items(), key=lambda item: (-item[1][0], item[0]))[:top_k]
            for rank, (article_id, (score, chunk_id)) in enumerate(ordered, start=1):
                rows.append(
                    {
                        "query_id": query_id,
                        "article_id": article_id,
                        "score": score,
                        "rank": rank,
                        "best_chunk_id": chunk_id,
                    }
                )
        return pd.DataFrame(rows, columns=RESULT_COLUMNS) if rows else _empty_results()


class QueryKnnChannel:
    name = "query_knn"

    def __init__(
        self,
        neighbors: int = 50,
        frequency_penalty: float = 0.5,
        word_weight: float = 0.5,
        voting: str = "similarity_sum",
    ) -> None:
        if neighbors <= 0:
            raise ValueError("query-kNN neighbors must be positive")
        if frequency_penalty < 0:
            raise ValueError("query-kNN frequency penalty must be non-negative")
        if not 0 <= word_weight <= 1:
            raise ValueError("query-kNN word weight must be in [0, 1]")
        if voting not in {"similarity_sum", "count", "max"}:
            raise ValueError("query-kNN voting must be similarity_sum, count or max")
        self.neighbors = neighbors
        self.frequency_penalty = frequency_penalty
        self.word_weight = word_weight
        self.voting = voting

    def fit(self, calibration: pd.DataFrame) -> "QueryKnnChannel":
        if "ground_truth" not in calibration.columns:
            raise ValueError("query-kNN training data must contain ground_truth")
        self.query_texts = calibration["query_text"].astype(str).tolist()
        self.labels = [tuple(int(item) for item in value.split()) for value in calibration["ground_truth"]]
        self.label_frequency = Counter(
            article_id for labels in self.labels for article_id in labels
        )
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(self.query_texts)
        self.char_matrix = self.char_vectorizer.fit_transform(self.query_texts)
        return self

    def search(self, queries: pd.DataFrame, top_k: int) -> pd.DataFrame:
        if not self.query_texts:
            return _empty_results()
        word_queries = self.word_vectorizer.transform(queries["query_text"].astype(str))
        char_queries = self.char_vectorizer.transform(queries["query_text"].astype(str))
        similarities = self.word_weight * (word_queries @ self.word_matrix.T) + (
            1.0 - self.word_weight
        ) * (char_queries @ self.char_matrix.T)
        rows: list[dict[str, object]] = []
        for row_index, query_id in enumerate(queries["query_id"].astype(int)):
            scores = similarities.getrow(row_index).toarray().ravel()
            positive = np.flatnonzero(scores > 0)
            neighbor_order = positive[
                np.lexsort((positive, -scores[positive]))[: self.neighbors]
            ]
            article_scores: dict[int, float] = {}
            for neighbor_index in neighbor_order:
                similarity = float(scores[neighbor_index])
                for article_id in self.labels[neighbor_index]:
                    if self.voting == "count":
                        article_scores[article_id] = article_scores.get(article_id, 0.0) + 1.0
                    elif self.voting == "max":
                        article_scores[article_id] = max(
                            article_scores.get(article_id, 0.0), similarity
                        )
                    else:
                        article_scores[article_id] = article_scores.get(article_id, 0.0) + similarity
            for article_id in article_scores:
                article_scores[article_id] /= self.label_frequency[article_id] ** self.frequency_penalty
            ordered = sorted(article_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            for rank, (article_id, score) in enumerate(ordered, start=1):
                rows.append(
                    {
                        "query_id": query_id,
                        "article_id": article_id,
                        "score": score,
                        "rank": rank,
                        "best_chunk_id": None,
                    }
                )
        return pd.DataFrame(rows, columns=RESULT_COLUMNS) if rows else _empty_results()
