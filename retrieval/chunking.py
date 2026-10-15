from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from preprocessing import normalize_text


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...


@dataclass(frozen=True, slots=True)
class _Piece:
    text: str
    token_ids: tuple[int, ...]
    source_block_ids: tuple[str, ...]


CHUNK_COLUMNS = [
    "chunk_id",
    "article_id",
    "position",
    "section_path",
    "clean_text",
    "normalized_text",
    "dense_text",
    "source_block_ids",
]


def build_retrieval_chunks(
    articles: pd.DataFrame,
    blocks: pd.DataFrame,
    tokenizer: Tokenizer,
    max_content_tokens: int = 384,
    overlap_tokens: int = 64,
    passage_prefix: str = "passage: ",
) -> pd.DataFrame:
    if not 0 <= overlap_tokens < max_content_tokens:
        raise ValueError("overlap_tokens must be smaller than max_content_tokens")
    titles = dict(zip(articles["article_id"].astype(int), articles["title"].astype(str)))
    rows: list[dict[str, object]] = []

    for article_id, article_blocks in blocks.groupby("article_id", sort=False):
        article_id = int(article_id)
        title = titles[article_id]
        position = 0
        previous_path: tuple[str, ...] | None = None
        group: list[_Piece] = []

        def flush_group() -> None:
            nonlocal position, group
            if not group or previous_path is None:
                group = []
                return
            for text, source_ids in _pack_pieces(group, max_content_tokens, overlap_tokens):
                path_text = "\n".join(previous_path)
                dense_parts = [f"{passage_prefix}{title}"]
                if path_text:
                    dense_parts.append(path_text)
                dense_parts.append(text)
                rows.append(
                    {
                        "chunk_id": f"{article_id}:{position:04d}",
                        "article_id": article_id,
                        "position": position,
                        "section_path": list(previous_path),
                        "clean_text": text,
                        "normalized_text": normalize_text(text),
                        "dense_text": "\n".join(dense_parts),
                        "source_block_ids": list(source_ids),
                    }
                )
                position += 1
            group = []

        for block in article_blocks.itertuples(index=False):
            section_path = tuple(str(item) for item in block.section_path)
            if previous_path is not None and section_path != previous_path:
                flush_group()
            previous_path = section_path
            token_ids = _encode_tokens(tokenizer, str(block.clean_text))
            if not token_ids:
                continue
            if len(token_ids) <= max_content_tokens:
                group.append(
                    _Piece(
                        text=str(block.clean_text),
                        token_ids=tuple(token_ids),
                        source_block_ids=(str(block.block_id),),
                    )
                )
                continue

            flush_group()
            step = max_content_tokens - overlap_tokens
            for start in range(0, len(token_ids), step):
                window = token_ids[start : start + max_content_tokens]
                if not window:
                    break
                text = tokenizer.decode(window, skip_special_tokens=True).strip()
                group.append(
                    _Piece(
                        text=text,
                        token_ids=tuple(window),
                        source_block_ids=(str(block.block_id),),
                    )
                )
                flush_group()
                if start + max_content_tokens >= len(token_ids):
                    break
        flush_group()

    return pd.DataFrame(rows, columns=CHUNK_COLUMNS)


def _encode_tokens(tokenizer: Tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(
            text,
            add_special_tokens=False,
            verbose=False,
        )
    except TypeError:
        return tokenizer.encode(text, add_special_tokens=False)


def _pack_pieces(
    pieces: list[_Piece],
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, tuple[str, ...]]]:
    packed: list[tuple[str, tuple[str, ...]]] = []
    current: list[_Piece] = []
    current_tokens = 0

    def emit() -> None:
        if not current:
            return
        packed.append(
            (
                "\n".join(piece.text for piece in current),
                tuple(
                    block_id
                    for piece in current
                    for block_id in piece.source_block_ids
                ),
            )
        )

    for piece in pieces:
        piece_size = len(piece.token_ids)
        if current and current_tokens + piece_size > max_tokens:
            emit()
            overlap: list[_Piece] = []
            overlap_size = 0
            for previous in reversed(current):
                if overlap_size + len(previous.token_ids) > overlap_tokens:
                    break
                overlap.insert(0, previous)
                overlap_size += len(previous.token_ids)
            while overlap and overlap_size + piece_size > max_tokens:
                removed = overlap.pop(0)
                overlap_size -= len(removed.token_ids)
            current = overlap
            current_tokens = overlap_size
        current.append(piece)
        current_tokens += piece_size
    emit()
    return packed
