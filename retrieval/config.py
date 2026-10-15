from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PathConfig:
    articles: Path
    blocks: Path
    calibration: Path
    test: Path
    cache_dir: Path


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    revision: str
    chunk_tokenizer_id: str
    chunk_tokenizer_revision: str
    query_prefix: str
    passage_prefix: str
    trust_remote_code: bool
    remote_code_revision: str | None
    max_parameters: int
    batch_size_gpu: int
    batch_size_cpu: int
    local_files_only: bool


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    max_content_tokens: int
    overlap_tokens: int


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    candidate_top_k: int
    chunk_top_k: int
    knn_neighbors: int
    knn_frequency_penalty: float
    knn_word_weight: float
    knn_voting: str
    bm25_k1: float
    bm25_b: float
    char_min_df: int


@dataclass(frozen=True, slots=True)
class FusionConfig:
    method: str
    score_normalization: str | None
    rrf_k: int
    weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    folds: int
    seed: int


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    paths: PathConfig
    model: ModelConfig
    chunking: ChunkConfig
    retrieval: RetrievalConfig
    fusion: FusionConfig
    evaluation: EvaluationConfig


def load_config(path: Path) -> CandidateGenerationConfig:
    with path.open("rb") as source:
        raw = tomllib.load(source)
    base = path.parent.parent.resolve()

    def project_path(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else base / candidate

    paths = raw["paths"]
    model = raw["model"]
    chunking = raw["chunking"]
    retrieval = raw["retrieval"]
    fusion = raw["fusion"]
    evaluation = raw["evaluation"]
    config = CandidateGenerationConfig(
        paths=PathConfig(
            articles=project_path(paths["articles"]),
            blocks=project_path(paths["blocks"]),
            calibration=project_path(paths["calibration"]),
            test=project_path(paths["test"]),
            cache_dir=project_path(paths["cache_dir"]),
        ),
        model=ModelConfig(
            model_id=str(model["model_id"]),
            revision=str(model["revision"]),
            chunk_tokenizer_id=str(
                model.get("chunk_tokenizer_id", model["model_id"])
            ),
            chunk_tokenizer_revision=str(
                model.get("chunk_tokenizer_revision", model["revision"])
            ),
            query_prefix=str(model.get("query_prefix", "query: ")),
            passage_prefix=str(model.get("passage_prefix", "passage: ")),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            remote_code_revision=(
                str(model["remote_code_revision"])
                if model.get("remote_code_revision")
                else None
            ),
            max_parameters=int(model["max_parameters"]),
            batch_size_gpu=int(model["batch_size_gpu"]),
            batch_size_cpu=int(model["batch_size_cpu"]),
            local_files_only=bool(model["local_files_only"]),
        ),
        chunking=ChunkConfig(
            max_content_tokens=int(chunking["max_content_tokens"]),
            overlap_tokens=int(chunking["overlap_tokens"]),
        ),
        retrieval=RetrievalConfig(
            candidate_top_k=int(retrieval["candidate_top_k"]),
            chunk_top_k=int(retrieval["chunk_top_k"]),
            knn_neighbors=int(retrieval["knn_neighbors"]),
            knn_frequency_penalty=float(retrieval["knn_frequency_penalty"]),
            knn_word_weight=float(retrieval.get("knn_word_weight", 0.5)),
            knn_voting=str(retrieval.get("knn_voting", "similarity_sum")),
            bm25_k1=float(retrieval["bm25_k1"]),
            bm25_b=float(retrieval["bm25_b"]),
            char_min_df=int(retrieval["char_min_df"]),
        ),
        fusion=FusionConfig(
            method=str(fusion.get("method", "rrf")),
            score_normalization=(
                str(fusion["score_normalization"])
                if fusion.get("score_normalization") is not None
                else None
            ),
            rrf_k=int(fusion["rrf_k"]),
            weights={key: float(value) for key, value in fusion["weights"].items()},
        ),
        evaluation=EvaluationConfig(
            folds=int(evaluation["folds"]),
            seed=int(evaluation["seed"]),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: CandidateGenerationConfig) -> None:
    if not 0 <= config.chunking.overlap_tokens < config.chunking.max_content_tokens:
        raise ValueError("chunk overlap must be non-negative and smaller than max tokens")
    if config.retrieval.candidate_top_k <= 0:
        raise ValueError("candidate_top_k must be positive")
    if config.model.max_parameters > 1_000_000_000:
        raise ValueError("configured model limit must not exceed 1B parameters")
    if config.model.trust_remote_code and not config.model.remote_code_revision:
        raise ValueError("trusted remote code requires a pinned remote_code_revision")
    if not 0 <= config.retrieval.knn_word_weight <= 1:
        raise ValueError("query-kNN word weight must be in [0, 1]")
    if config.retrieval.knn_voting not in {"similarity_sum", "count", "max"}:
        raise ValueError("unsupported query-kNN voting method")
    if config.fusion.method not in {"rrf", "linear"}:
        raise ValueError("fusion method must be 'rrf' or 'linear'")
    if config.fusion.method == "linear" and config.fusion.score_normalization not in {
        "max",
        "minmax",
    }:
        raise ValueError("linear fusion requires max or minmax score normalization")
    missing_weights = {
        "bm25_chunk",
        "char_tfidf",
        "dense",
        "query_knn",
    } - set(config.fusion.weights)
    if missing_weights:
        raise ValueError(f"missing fusion weights: {sorted(missing_weights)}")
