from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from retrieval.channels import (
    BM25ChunkChannel,
    CharTfidfChannel,
    DenseChunkChannel,
)
from retrieval.chunking import build_retrieval_chunks
from retrieval.config import CandidateGenerationConfig
from retrieval.dense import DenseEncoder
from retrieval.io import read_dataframe, write_dataframe, write_json


@dataclass(slots=True)
class UniversalIndex:
    articles: pd.DataFrame
    chunks: pd.DataFrame
    bm25_chunk: BM25ChunkChannel
    char_tfidf: CharTfidfChannel
    dense_embeddings: np.ndarray
    cache_key: str
    cache_dir: Path

    def channels(self, encoder: DenseEncoder) -> dict[str, object]:
        dense = DenseChunkChannel(
            encoder=encoder,
            chunk_top_k=self.bm25_chunk.chunk_top_k,
        ).fit(self.chunks, self.dense_embeddings)
        return {
            "bm25_chunk": self.bm25_chunk,
            "char_tfidf": self.char_tfidf,
            "dense": dense,
        }


def build_or_load_universal_index(
    config: CandidateGenerationConfig,
    encoder: DenseEncoder,
    force: bool = False,
) -> UniversalIndex:
    cache_key = compute_cache_key(config)
    cache_dir = config.paths.cache_dir / cache_key
    index_path = cache_dir / "universal_index.joblib"
    embeddings_path = cache_dir / "dense_embeddings.npy"
    chunks_path = cache_dir / "retrieval_chunks.f"
    manifest_path = cache_dir / "manifest.json"

    if not force and all(
        path.exists()
        for path in (index_path, embeddings_path, chunks_path, manifest_path)
    ):
        payload = joblib.load(index_path)
        return UniversalIndex(
            articles=payload["articles"],
            chunks=read_dataframe(chunks_path),
            bm25_chunk=payload["bm25_chunk"],
            char_tfidf=payload["char_tfidf"],
            dense_embeddings=np.load(embeddings_path, mmap_mode="r"),
            cache_key=cache_key,
            cache_dir=cache_dir,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    articles = read_dataframe(config.paths.articles)
    blocks = read_dataframe(config.paths.blocks)
    chunks = build_retrieval_chunks(
        articles=articles,
        blocks=blocks,
        tokenizer=encoder.tokenizer,
        max_content_tokens=config.chunking.max_content_tokens,
        overlap_tokens=config.chunking.overlap_tokens,
        passage_prefix=config.model.passage_prefix,
    )
    retrieval = config.retrieval
    bm25_chunk = BM25ChunkChannel(
        retrieval.bm25_k1,
        retrieval.bm25_b,
        chunk_top_k=retrieval.chunk_top_k,
    ).fit(articles, chunks)
    char_tfidf = CharTfidfChannel(min_df=retrieval.char_min_df).fit(articles)
    dense_embeddings = encoder.encode(chunks["dense_text"].astype(str).tolist())

    write_dataframe(chunks, chunks_path)
    np.save(embeddings_path, dense_embeddings)
    joblib.dump(
        {
            "articles": articles,
            "bm25_chunk": bm25_chunk,
            "char_tfidf": char_tfidf,
        },
        index_path,
        compress=3,
    )
    write_json(
        {
            "cache_key": cache_key,
            "model": asdict(config.model),
            "chunking": asdict(config.chunking),
            "retrieval": asdict(config.retrieval),
            "parameter_count": encoder.parameter_count,
            "device": encoder.device,
            "articles": len(articles),
            "chunks": len(chunks),
            "embedding_shape": list(dense_embeddings.shape),
            "python": platform.python_version(),
        },
        manifest_path,
    )
    return UniversalIndex(
        articles=articles,
        chunks=chunks,
        bm25_chunk=bm25_chunk,
        char_tfidf=char_tfidf,
        dense_embeddings=dense_embeddings,
        cache_key=cache_key,
        cache_dir=cache_dir,
    )


def compute_cache_key(config: CandidateGenerationConfig) -> str:
    digest = hashlib.sha256()
    for path in (config.paths.articles, config.paths.blocks):
        digest.update(_file_sha256(path).encode("ascii"))
    payload = {
        "model": asdict(config.model),
        "chunking": asdict(config.chunking),
        "retrieval": {
            "bm25_k1": config.retrieval.bm25_k1,
            "bm25_b": config.retrieval.bm25_b,
            "char_min_df": config.retrieval.char_min_df,
            "chunk_top_k": config.retrieval.chunk_top_k,
        },
    }
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
