from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa

from preprocessing.html import preprocess_html
from preprocessing.models import ProcessedArticle, TextBlock


REQUIRED_ARTICLE_COLUMNS = {"article_id", "title", "body"}
ARTICLE_SCHEMA = pa.schema(
    [
        pa.field("article_id", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("clean_text", pa.string(), nullable=False),
        pa.field("normalized_text", pa.string(), nullable=False),
        pa.field("outgoing_article_ids", pa.list_(pa.int64()), nullable=False),
    ]
)
BLOCK_SCHEMA = pa.schema(
    [
        pa.field("block_id", pa.string(), nullable=False),
        pa.field("article_id", pa.int64(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("block_type", pa.string(), nullable=False),
        pa.field("section_path", pa.list_(pa.string()), nullable=False),
        pa.field("clean_text", pa.string(), nullable=False),
        pa.field("normalized_text", pa.string(), nullable=False),
        pa.field("outgoing_article_ids", pa.list_(pa.int64()), nullable=False),
    ]
)


def preprocess_articles(
    articles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_columns = REQUIRED_ARTICLE_COLUMNS - set(articles.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required article columns: {missing}")

    article_ids = articles["article_id"].astype("int64")
    if article_ids.duplicated().any():
        duplicate_ids = sorted(article_ids[article_ids.duplicated(keep=False)].unique())
        raise ValueError(f"Duplicate article_id values: {duplicate_ids}")
    valid_article_ids = frozenset(int(article_id) for article_id in article_ids)

    processed_articles: list[ProcessedArticle] = []
    processed_blocks: list[TextBlock] = []
    for row in articles.itertuples(index=False):
        article, blocks = preprocess_html(
            article_id=int(row.article_id),
            title="" if row.title is None else str(row.title),
            body_html="" if row.body is None else str(row.body),
            valid_article_ids=valid_article_ids,
        )
        processed_articles.append(article)
        processed_blocks.extend(blocks)

    return (
        _articles_dataframe(processed_articles),
        _blocks_dataframe(processed_blocks),
    )


def write_preprocessed_feather(
    articles: pd.DataFrame,
    blocks: pd.DataFrame,
    articles_output: Path,
    blocks_output: Path,
) -> None:
    articles_output.parent.mkdir(parents=True, exist_ok=True)
    blocks_output.parent.mkdir(parents=True, exist_ok=True)
    _write_arrow_ipc(
        pa.Table.from_pandas(
            articles,
            schema=ARTICLE_SCHEMA,
            preserve_index=False,
            safe=True,
        ),
        articles_output,
    )
    _write_arrow_ipc(
        pa.Table.from_pandas(
            blocks,
            schema=BLOCK_SCHEMA,
            preserve_index=False,
            safe=True,
        ),
        blocks_output,
    )


def read_feather_dataframe(path: Path) -> pd.DataFrame:
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).read_all().to_pandas()


def _write_arrow_ipc(table: pa.Table, path: Path) -> None:
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _articles_dataframe(articles: Iterable[ProcessedArticle]) -> pd.DataFrame:
    rows = [
        {
            "article_id": article.article_id,
            "title": article.title,
            "clean_text": article.clean_text,
            "normalized_text": article.normalized_text,
            "outgoing_article_ids": list(article.outgoing_article_ids),
        }
        for article in articles
    ]
    return pd.DataFrame(rows, columns=ARTICLE_SCHEMA.names)


def _blocks_dataframe(blocks: Iterable[TextBlock]) -> pd.DataFrame:
    rows = [
        {
            "block_id": block.block_id,
            "article_id": block.article_id,
            "position": block.position,
            "block_type": block.block_type,
            "section_path": list(block.section_path),
            "clean_text": block.clean_text,
            "normalized_text": block.normalized_text,
            "outgoing_article_ids": list(block.outgoing_article_ids),
        }
        for block in blocks
    ]
    return pd.DataFrame(rows, columns=BLOCK_SCHEMA.names)
