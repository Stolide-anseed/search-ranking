from __future__ import annotations

from pathlib import Path

import pandas as pd

from reranking.zero_shot import (
    cross_encoder_cache_identity,
    load_config as load_reranker_config,
    score_cross_encoder,
    validate_score_cache,
)
from retrieval.channels import QueryKnnChannel
from retrieval.config import load_config
from retrieval.dense import DenseEncoder
from retrieval.hybrid import HybridCandidateGenerator
from retrieval.index import build_or_load_universal_index
from retrieval.io import read_dataframe, write_dataframe
from submission import InferenceConfig


def rebuild_production_scores(
    root: Path,
    force_index: bool = False,
    reprocess: bool = False,
    inference_config: InferenceConfig | None = None,
    manifest: dict[str, object] | None = None,
    force_model: bool = False,
) -> pd.DataFrame:
    if inference_config is None:
        raise ValueError("inference_config from artifacts/manifest.json is required")
    if manifest is None:
        raise ValueError("artifacts/manifest.json payload is required")
    if reprocess:
        from preprocessing import preprocess_articles, write_preprocessed_feather

        raw_articles = read_dataframe(root / "data" / "articles.f")
        processed_articles, article_blocks = preprocess_articles(raw_articles)
        write_preprocessed_feather(
            processed_articles,
            article_blocks,
            root / "artifacts" / "processed_articles.f",
            root / "artifacts" / "article_blocks.f",
        )
        print(
            f"Preprocessed {len(processed_articles)} articles into "
            f"{len(article_blocks)} blocks"
        )
    candidate_config = load_config(root / "configs" / "candidate_generation.toml")
    if candidate_config.retrieval.candidate_top_k != inference_config.candidate_top_k:
        raise ValueError("candidate_top_k differs between manifest and retrieval config")
    encoder = DenseEncoder(candidate_config.model).load()
    index = build_or_load_universal_index(candidate_config, encoder, force=force_index)
    calibration = read_dataframe(candidate_config.paths.calibration)
    test_queries = read_dataframe(candidate_config.paths.test)
    query_knn = QueryKnnChannel(
        neighbors=candidate_config.retrieval.knn_neighbors,
        frequency_penalty=candidate_config.retrieval.knn_frequency_penalty,
        word_weight=candidate_config.retrieval.knn_word_weight,
        voting=candidate_config.retrieval.knn_voting,
    ).fit(calibration)
    channels = {**index.channels(encoder), "query_knn": query_knn}
    generator = HybridCandidateGenerator(
        article_ids=index.articles["article_id"].astype(int).tolist(),
        channels=channels,
        weights=candidate_config.fusion.weights,
        rrf_k=candidate_config.fusion.rrf_k,
        method=candidate_config.fusion.method,
        score_normalization=candidate_config.fusion.score_normalization,
        channel_top_k={name: len(index.articles) for name in channels},
    )
    candidates = generator.generate(
        test_queries,
        top_k=inference_config.candidate_top_k,
    )
    reranker_input = candidates[
        candidates["candidate_rank"] <= inference_config.rerank_depth
    ].copy()
    enriched = add_reranker_text(
        reranker_input,
        test_queries,
        index.articles,
        index.chunks,
    )
    reranker_config = load_reranker_config(root / "configs" / "reranker.toml")
    cache_identity = cross_encoder_cache_identity(enriched, reranker_config)
    if (
        not force_model
        and cache_identity == manifest.get("production_score_cache_identity")
    ):
        bundled = read_dataframe(root / "artifacts" / "production_scores.f")
        scores = bundled.loc[
            bundled["candidate_rank"] <= inference_config.rerank_depth,
            ["query_id", "article_id", "cross_encoder_score"],
        ].copy()
        validate_score_cache(scores, enriched)
        print("GTE reranker: reused validated production score cache")
    else:
        scores, parameter_count, device = score_cross_encoder(enriched, reranker_config)
        print(f"GTE reranker: {parameter_count:,} parameters, device={device}")
    compact = candidates.merge(
        scores[["query_id", "article_id", "cross_encoder_score"]],
        on=["query_id", "article_id"],
        how="left",
        validate="one_to_one",
    )
    compact = compact[
        ["query_id", "article_id", "candidate_rank", "fusion_score", "cross_encoder_score"]
    ].copy()
    write_dataframe(compact, root / "artifacts" / "rebuilt_production_scores.f")
    return compact


def add_reranker_text(
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    articles: pd.DataFrame,
    chunks: pd.DataFrame,
) -> pd.DataFrame:
    result = candidates.copy()
    query_text = queries.set_index("query_id")["query_text"]
    article_title = articles.set_index("article_id")["title"]
    chunk_text = chunks.set_index("chunk_id")["clean_text"]
    result.insert(1, "query_text", result["query_id"].map(query_text).fillna(""))
    result.insert(4, "article_title", result["article_id"].map(article_title).fillna(""))
    bm25_column = (
        "best_bm25_chunk_id"
        if "best_bm25_chunk_id" in result.columns
        else "best_bm25_chunk_chunk_id"
    )
    result["bm25_chunk_text"] = result[bm25_column].map(chunk_text).fillna("")
    result["dense_text"] = result["best_dense_chunk_id"].map(chunk_text).fillna("")
    return result
