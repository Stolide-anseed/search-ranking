from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "BM25ChunkChannel": ("retrieval.channels", "BM25ChunkChannel"),
    "CandidateChannel": ("retrieval.channels", "CandidateChannel"),
    "CandidateGenerationConfig": ("retrieval.config", "CandidateGenerationConfig"),
    "CharTfidfChannel": ("retrieval.channels", "CharTfidfChannel"),
    "DenseChunkChannel": ("retrieval.channels", "DenseChunkChannel"),
    "HybridCandidateGenerator": ("retrieval.hybrid", "HybridCandidateGenerator"),
    "QueryKnnChannel": ("retrieval.channels", "QueryKnnChannel"),
    "UniversalIndex": ("retrieval.index", "UniversalIndex"),
    "build_or_load_universal_index": ("retrieval.index", "build_or_load_universal_index"),
    "build_retrieval_chunks": ("retrieval.chunking", "build_retrieval_chunks"),
    "fuse_channel_results": ("retrieval.hybrid", "fuse_channel_results"),
    "load_config": ("retrieval.config", "load_config"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
