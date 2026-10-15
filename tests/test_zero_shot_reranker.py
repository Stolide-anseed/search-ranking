from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reranking.zero_shot import (
    analyze_regressions,
    build_cross_encoder_views,
    cross_encoder_cache_identity,
    cross_encoder_model_kwargs,
    download_remote_code,
    enforce_parameter_limit,
    fast_ranking_metrics,
    load_config,
    rerank_candidates,
    score_cache_matches,
)


PROJECT_ROOT = Path(__file__).parents[1]


def test_cross_encoder_views_deduplicate_equal_chunks() -> None:
    candidates = pd.DataFrame(
        {
            "query_text": ["query", "query"],
            "article_title": ["title", "title 2"],
            "bm25_chunk_text": ["same text", "bm25"],
            "dense_text": [" Same   text ", "dense"],
        }
    )

    pairs, indices, sources = build_cross_encoder_views(candidates)

    assert len(pairs) == 3
    assert indices.tolist() == [0, 1, 1]
    assert sources == ["bm25", "bm25", "dense"]


def test_rerank_candidates_only_reorders_selected_depth() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1],
            "article_id": [10, 20, 30, 40],
            "candidate_rank": [1, 2, 3, 4],
            "fusion_score": [4.0, 3.0, 2.0, 1.0],
            "cross_encoder_score": [0.1, 0.9, 1.0, 100.0],
        }
    )

    result = rerank_candidates(candidates, depth=3, alpha=1.0)

    assert result["article_id"].tolist() == [30, 20, 10, 40]
    assert result["candidate_rank"].tolist() == [1, 2, 3, 4]


def test_parameter_limit_is_enforced() -> None:
    enforce_parameter_limit(999_999_999, 1_000_000_000)
    with pytest.raises(ValueError, match="limit"):
        enforce_parameter_limit(1_000_000_001, 1_000_000_000)


def test_fast_metrics_use_full_ground_truth_denominator() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": [1, 1],
            "article_id": [10, 30],
            "candidate_rank": [1, 2],
            "fusion_score": [1.0, 0.0],
            "cross_encoder_score": [1.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {"query_id": [1], "query_text": ["q"], "ground_truth": ["10 20"]}
    )

    metrics = fast_ranking_metrics(candidates, truth, depth=2, alpha=0.5)

    assert metrics["map_at_10"] == 0.5
    assert metrics["recall_at_10"] == 0.5


def test_fast_metrics_include_unchanged_tail_for_shallow_depth() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": [1] * 10,
            "article_id": list(range(1, 11)),
            "candidate_rank": list(range(1, 11)),
            "fusion_score": list(reversed(range(1, 11))),
            "cross_encoder_score": list(reversed(range(1, 11))),
        }
    )
    truth = pd.DataFrame(
        {"query_id": [1], "query_text": ["q"], "ground_truth": ["5"]}
    )

    fast = fast_ranking_metrics(candidates, truth, depth=1, alpha=1.0)
    actual = rerank_candidates(candidates, depth=1, alpha=1.0)

    assert fast["map_at_10"] == 0.2
    assert fast["map_at_10"] == pytest.approx(
        1 / int(actual.loc[actual["article_id"] == 5, "candidate_rank"].iloc[0])
    )


def test_fast_metrics_count_duplicate_relevant_article_once() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": [1, 1],
            "article_id": [10, 10],
            "candidate_rank": [1, 2],
            "fusion_score": [1.0, 0.0],
            "cross_encoder_score": [1.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {"query_id": [1], "query_text": ["q"], "ground_truth": ["10"]}
    )

    metrics = fast_ranking_metrics(candidates, truth, depth=2, alpha=0.5)

    assert metrics["map_at_10"] == 1.0
    assert metrics["recall_at_10"] == 1.0


def test_regression_analysis_detects_lost_second_relevant() -> None:
    rows = []
    for rank, article_id in enumerate([10, 20, 30], start=1):
        rows.append(
            {
                "query_id": 1,
                "article_id": article_id,
                "candidate_rank": rank,
                "article_title": f"article {article_id}",
                "bm25_chunk_text": "text",
                "dense_text": "text",
            }
        )
    baseline = pd.DataFrame(rows)
    reranked = baseline.set_index("article_id").loc[[10, 30, 20]].reset_index()
    reranked["candidate_rank"] = [1, 2, 3]
    truth = pd.DataFrame(
        {"query_id": [1], "query_text": ["query"], "ground_truth": ["10 20"]}
    )

    regressions = analyze_regressions(baseline, reranked, truth)

    assert len(regressions) == 1
    assert regressions.iloc[0]["category"] == "loses_second_relevant"
    assert regressions.iloc[0]["relevant_rank_changes"] == "10:1->1; 20:2->3"


def test_regression_analysis_returns_typed_empty_frame() -> None:
    baseline = pd.DataFrame(
        {
            "query_id": [1],
            "article_id": [10],
            "candidate_rank": [1],
            "article_title": ["article"],
            "bm25_chunk_text": ["text"],
            "dense_text": ["text"],
        }
    )
    truth = pd.DataFrame(
        {"query_id": [1], "query_text": ["query"], "ground_truth": ["10"]}
    )

    regressions = analyze_regressions(baseline, baseline.copy(), truth)

    assert regressions.empty
    assert {"query_id", "ap_delta", "category"} <= set(regressions.columns)


def test_cross_encoder_cache_identity_covers_model_and_text(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "reranker.toml")
    candidates = pd.DataFrame(
        {
            "query_id": [1],
            "article_id": [10],
            "query_text": ["query"],
            "article_title": ["title"],
            "bm25_chunk_text": ["bm25"],
            "dense_text": ["dense"],
        }
    )
    identity = cross_encoder_cache_identity(candidates, config)
    changed_text = candidates.assign(dense_text="changed")
    changed_model = replace(config, revision="different")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"cache_identity": identity}), encoding="utf-8"
    )

    assert score_cache_matches(manifest, identity)
    assert cross_encoder_cache_identity(changed_text, config) != identity
    assert cross_encoder_cache_identity(candidates, changed_model) != identity


def test_default_config_exists_and_loads() -> None:
    config = load_config()
    assert config.oof_candidates.exists()
    assert config.remote_code_revision == "40ced75c3017eb27626c9d4ea981bde21a2662f4"


def test_remote_code_download_uses_pinned_revision() -> None:
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return Path(kwargs["revision"])

    config = replace(
        load_config(PROJECT_ROOT / "configs" / "reranker.toml"),
        local_files_only=True,
    )
    result = download_remote_code(fake_download, config)

    assert result.name == "40ced75c3017eb27626c9d4ea981bde21a2662f4"
    assert calls[0]["revision"] == result.name
    assert calls[0]["repo_id"] == "Alibaba-NLP/new-impl"
    assert cross_encoder_model_kwargs(config)["code_revision"] == result.name
