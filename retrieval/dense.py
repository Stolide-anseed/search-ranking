from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from retrieval.config import ModelConfig


@dataclass(slots=True)
class DenseEncoder:
    config: ModelConfig
    device: str = field(init=False, default="cpu")
    model: Any = field(init=False, default=None, repr=False)
    tokenizer: Any = field(init=False, default=None, repr=False)
    parameter_count: int = field(init=False, default=0)

    def load(self) -> "DenseEncoder":
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        self.device = select_device(torch.cuda.is_available())
        self.model = SentenceTransformer(
            self.config.model_id,
            revision=self.config.revision,
            device=self.device,
            local_files_only=self.config.local_files_only,
            trust_remote_code=self.config.trust_remote_code,
            model_kwargs=(
                {"code_revision": self.config.remote_code_revision}
                if self.config.remote_code_revision
                else None
            ),
        )
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        try:
            enforce_parameter_limit(
                self.parameter_count,
                self.config.max_parameters,
            )
        except ValueError:
            del self.model
            raise
        repair_gte_nonpersistent_buffers(self.model)
        if (
            self.config.chunk_tokenizer_id == self.config.model_id
            and self.config.chunk_tokenizer_revision == self.config.revision
        ):
            self.tokenizer = self.model.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.chunk_tokenizer_id,
                revision=self.config.chunk_tokenizer_revision,
                local_files_only=self.config.local_files_only,
                trust_remote_code=False,
            )
        return self

    @property
    def batch_size(self) -> int:
        return (
            self.config.batch_size_gpu
            if getattr(self, "device", "cpu") == "cuda"
            else self.config.batch_size_cpu
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) >= 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self.encode([f"{self.config.query_prefix}{query}" for query in queries])


def repair_gte_nonpersistent_buffers(sentence_model: Any) -> None:
    """Repair meta/nonpersistent RoPE buffers in Alibaba's custom implementation."""
    import torch

    transformer = sentence_model[0]
    core = getattr(transformer, "auto_model", None)
    if core is None:
        return
    base_model = core.new if hasattr(core, "new") else core
    if not hasattr(base_model, "embeddings"):
        return
    embeddings = base_model.embeddings
    if not hasattr(embeddings, "rotary_emb"):
        return
    device = embeddings.word_embeddings.weight.device
    embeddings.position_ids = torch.arange(
        core.config.max_position_embeddings,
        device=device,
    )
    rotary = embeddings.rotary_emb
    embeddings.rotary_emb = type(rotary)(
        dim=rotary.dim,
        max_position_embeddings=rotary.max_position_embeddings,
        base=rotary.base,
        scaling_factor=rotary.scaling_factor,
        mixed_b=rotary.mixed_b,
    ).to(device)


def select_device(cuda_available: bool) -> str:
    return "cuda" if cuda_available else "cpu"


def enforce_parameter_limit(parameter_count: int, limit: int) -> None:
    if parameter_count > limit:
        raise ValueError(
            f"Model has {parameter_count:,} parameters; limit is {limit:,}"
        )
