from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from preprocessing import normalize_text


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(value: object) -> list[str]:
    return TOKEN_RE.findall(normalize_text(value))


@dataclass(slots=True)
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    document_count: int = field(init=False, default=0)
    document_lengths: np.ndarray = field(
        init=False,
        default_factory=lambda: np.asarray([], dtype=np.float32),
    )
    average_length: float = field(init=False, default=0.0)
    postings: dict[str, tuple[np.ndarray, np.ndarray, float]] = field(
        init=False,
        default_factory=dict,
    )

    def fit(self, documents: Sequence[str]) -> "BM25Index":
        tokenized = [tokenize(document) for document in documents]
        self.document_count = len(tokenized)
        self.document_lengths = np.asarray(
            [len(tokens) for tokens in tokenized], dtype=np.float32
        )
        self.average_length = float(self.document_lengths.mean()) if tokenized else 0.0
        postings: dict[str, list[tuple[int, int]]] = {}
        for document_index, tokens in enumerate(tokenized):
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            for token, frequency in frequencies.items():
                postings.setdefault(token, []).append((document_index, frequency))

        self.postings: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for token, values in postings.items():
            document_indices = np.asarray([value[0] for value in values], dtype=np.int32)
            term_frequencies = np.asarray([value[1] for value in values], dtype=np.float32)
            document_frequency = len(values)
            inverse_document_frequency = math.log(
                1.0
                + (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            self.postings[token] = (
                document_indices,
                term_frequencies,
                inverse_document_frequency,
            )
        return self

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(self.document_count, dtype=np.float32)
        if self.document_count == 0 or self.average_length == 0:
            return result
        for token in set(tokenize(query)):
            posting = self.postings.get(token)
            if posting is None:
                continue
            document_indices, term_frequencies, inverse_document_frequency = posting
            lengths = self.document_lengths[document_indices]
            denominator = term_frequencies + self.k1 * (
                1.0 - self.b + self.b * lengths / self.average_length
            )
            result[document_indices] += inverse_document_frequency * (
                term_frequencies * (self.k1 + 1.0) / denominator
            )
        return result

    def search(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.scores(query)
        positive = np.flatnonzero(scores > 0)
        if len(positive) == 0:
            return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)
        order = np.lexsort((positive, -scores[positive]))[:top_k]
        indices = positive[order]
        return indices, scores[indices]
