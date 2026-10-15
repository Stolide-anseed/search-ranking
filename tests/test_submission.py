from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from submission import (
    InferenceConfig,
    build_answer,
    load_manifest,
    minmax,
    validate_scores,
)


CONFIG = InferenceConfig(candidate_top_k=100, rerank_depth=30, alpha=0.25)


def test_minmax() -> None:
    assert np.allclose(minmax(pd.Series([2.0, 4.0, 3.0])), [0.0, 1.0, 0.5])
    assert np.allclose(minmax(pd.Series([2.0, 2.0])), [1.0, 1.0])


def test_build_answer_is_stable_and_valid() -> None:
    query = pd.DataFrame({"query_id": [7], "query_text": ["test"]})
    articles = pd.DataFrame({"article_id": range(1, 101)})
    scores = pd.DataFrame(
        {
            "query_id": [7] * 100,
            "article_id": range(1, 101),
            "candidate_rank": range(1, 101),
            "fusion_score": np.arange(100, 0, -1, dtype=float),
            "cross_encoder_score": [0.0] * CONFIG.rerank_depth
            + [np.nan] * (100 - CONFIG.rerank_depth),
        }
    )
    answer = build_answer(scores, query, articles, CONFIG)
    assert answer.columns.tolist() == ["query_id", "answer"]
    assert answer.iloc[0]["answer"] == "1 2 3 4 5 6 7 8 9 10"


def test_validate_scores_rejects_duplicate_pairs() -> None:
    query = pd.DataFrame({"query_id": [1]})
    articles = pd.DataFrame({"article_id": range(1, 101)})
    scores = pd.DataFrame(
        {
            "query_id": [1] * 100,
            "article_id": [1] * 100,
            "candidate_rank": range(1, 101),
            "fusion_score": np.ones(100),
            "cross_encoder_score": [1.0] * CONFIG.rerank_depth
            + [np.nan] * (100 - CONFIG.rerank_depth),
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_scores(scores, query, articles, CONFIG)


def test_tail_must_not_have_cross_encoder_scores() -> None:
    query = pd.DataFrame({"query_id": [1]})
    articles = pd.DataFrame({"article_id": range(1, 101)})
    scores = pd.DataFrame(
        {
            "query_id": [1] * 100,
            "article_id": range(1, 101),
            "candidate_rank": range(1, 101),
            "fusion_score": np.ones(100),
            "cross_encoder_score": [1.0] * (CONFIG.rerank_depth + 1)
            + [np.nan] * (99 - CONFIG.rerank_depth),
        }
    )
    with pytest.raises(ValueError, match="below rerank_depth"):
        validate_scores(scores, query, articles, CONFIG)


def test_manifest_is_the_inference_config_source() -> None:
    _, config = load_manifest(Path(__file__).parents[1])
    assert config == CONFIG


def test_reranker_does_not_promote_unscored_tail() -> None:
    config = InferenceConfig(candidate_top_k=12, rerank_depth=10, alpha=1.0)
    query = pd.DataFrame({"query_id": [1]})
    articles = pd.DataFrame({"article_id": range(1, 13)})
    scores = pd.DataFrame(
        {
            "query_id": [1] * 12,
            "article_id": range(1, 13),
            "candidate_rank": range(1, 13),
            "fusion_score": [0.0] * 10 + [100.0, 99.0],
            "cross_encoder_score": list(range(10, 0, -1)) + [np.nan, np.nan],
        }
    )
    answer = build_answer(scores, query, articles, config)
    assert answer.iloc[0]["answer"] == "1 2 3 4 5 6 7 8 9 10"
