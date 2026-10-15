from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextBlock:
    block_id: str
    article_id: int
    position: int
    block_type: str
    section_path: tuple[str, ...]
    clean_text: str
    normalized_text: str
    outgoing_article_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProcessedArticle:
    article_id: int
    title: str
    clean_text: str
    normalized_text: str
    outgoing_article_ids: tuple[int, ...]
