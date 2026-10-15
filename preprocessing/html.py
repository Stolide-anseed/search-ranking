from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import AbstractSet
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

from preprocessing.models import ProcessedArticle, TextBlock


SPACE_RE = re.compile(r"\s+")
ARTICLE_PATH_RE = re.compile(r"(?:^|/)articles/(\d+)(?:/|$)")
HEADING_NAMES = {f"h{level}" for level in range(1, 7)}
INLINE_NAMES = {
    "a",
    "abbr",
    "b",
    "br",
    "cite",
    "code",
    "em",
    "i",
    "label",
    "mark",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "time",
}
REMOVED_NAMES = {"input", "noscript", "script", "style"}


def clean_text(value: object) -> str:
    text = html.unescape("" if value is None else str(value))
    text = unicodedata.normalize("NFC", text).replace("\xa0", " ")
    return SPACE_RE.sub(" ", text).strip()


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold().replace("ё", "е")
    text = "".join(character if character.isalnum() or character == "_" else " " for character in text)
    return SPACE_RE.sub(" ", text).strip()


@dataclass(slots=True)
class _DraftBlock:
    block_type: str
    section_path: tuple[str, ...]
    clean_text: str
    normalized_text: str
    outgoing_article_ids: set[int]


class _Extractor:
    def __init__(
        self,
        article_id: int,
        valid_article_ids: AbstractSet[int],
    ) -> None:
        self.article_id = article_id
        self.valid_article_ids = valid_article_ids
        self.blocks: list[_DraftBlock] = []

    def extract(self, body_html: str) -> list[TextBlock]:
        soup = BeautifulSoup(body_html, "lxml")
        for node in soup.find_all(REMOVED_NAMES):
            node.decompose()

        root = soup.body or soup
        self._walk_container(root, prefix=(), headings=[])
        drafts = self._deduplicate(self.blocks)
        return [
            TextBlock(
                block_id=f"{self.article_id}:{position:04d}",
                article_id=self.article_id,
                position=position,
                block_type=draft.block_type,
                section_path=draft.section_path,
                clean_text=draft.clean_text,
                normalized_text=draft.normalized_text,
                outgoing_article_ids=tuple(sorted(draft.outgoing_article_ids)),
            )
            for position, draft in enumerate(drafts)
        ]

    @staticmethod
    def _path(
        prefix: tuple[str, ...], headings: list[tuple[int, str]]
    ) -> tuple[str, ...]:
        return prefix + tuple(text for _, text in headings)

    def _emit(
        self,
        block_type: str,
        value: object,
        section_path: tuple[str, ...],
        links: Iterable[int] = (),
    ) -> None:
        cleaned = clean_text(value)
        normalized = normalize_text(cleaned)
        if not normalized:
            return
        self.blocks.append(
            _DraftBlock(
                block_type=block_type,
                section_path=section_path,
                clean_text=cleaned,
                normalized_text=normalized,
                outgoing_article_ids=set(links),
            )
        )

    def _walk_container(
        self,
        node: Tag | BeautifulSoup,
        prefix: tuple[str, ...],
        headings: list[tuple[int, str]],
        excluded_nodes: AbstractSet[int] = frozenset(),
    ) -> None:
        buffered_text: list[str] = []
        buffered_links: set[int] = set()

        def flush_buffer() -> None:
            if buffered_text:
                self._emit(
                    "text",
                    " ".join(buffered_text),
                    self._path(prefix, headings),
                    buffered_links,
                )
                buffered_text.clear()
                buffered_links.clear()

        for child in node.children:
            if id(child) in excluded_nodes:
                continue
            if isinstance(child, NavigableString):
                value = clean_text(child)
                if value:
                    buffered_text.append(value)
                continue
            if not isinstance(child, Tag):
                continue
            if child.name in REMOVED_NAMES:
                continue
            if child.name in INLINE_NAMES:
                value = self._node_text(child)
                if value:
                    buffered_text.append(value)
                    buffered_links.update(self._node_links(child))
                continue

            flush_buffer()
            self._visit(child, prefix, headings)

        flush_buffer()

    def _visit(
        self,
        node: Tag,
        prefix: tuple[str, ...],
        headings: list[tuple[int, str]],
    ) -> None:
        classes = set(node.get("class", ()))

        if node.name in HEADING_NAMES:
            self._handle_heading(node, prefix, headings)
        elif node.name == "headline":
            self._handle_headline(node, prefix, headings)
        elif node.name == "chunk":
            self._handle_chunk(node, prefix, headings)
        elif "tabset" in classes:
            self._handle_tabset(node, prefix, headings)
        elif "spoiler" in classes:
            self._handle_spoiler(node, prefix, headings)
        elif "factoid" in classes:
            self._handle_factoid(node, prefix, headings)
        elif node.name == "table":
            self._handle_table(node, prefix, headings)
        elif node.name == "p":
            self._emit(
                "paragraph",
                self._node_text(node),
                self._path(prefix, headings),
                self._node_links(node),
            )
        elif node.name == "li":
            self._handle_list_item(node, prefix, headings)
        elif "spoiler-text" in classes:
            self._handle_scoped_title("spoiler_title", node, prefix, headings)
        elif "tab-label" in classes:
            self._handle_scoped_title("tab_title", node, prefix, headings)
        elif node.name == "img":
            self._emit(
                "image_alt",
                node.get("alt", ""),
                self._path(prefix, headings),
            )
        else:
            self._walk_container(node, prefix, headings)

    def _handle_heading(
        self,
        node: Tag,
        prefix: tuple[str, ...],
        headings: list[tuple[int, str]],
    ) -> None:
        text = self._node_text(node)
        if not normalize_text(text):
            return
        level = int(node.name[1])
        while headings and headings[-1][0] >= level:
            headings.pop()
        headings.append((level, text))
        self._emit(
            "heading",
            text,
            self._path(prefix, headings),
            self._node_links(node),
        )

    def _handle_headline(
        self,
        node: Tag,
        prefix: tuple[str, ...],
        headings: list[tuple[int, str]],
    ) -> None:
        nested_heading = node.find(HEADING_NAMES)
        if nested_heading is not None:
            self._handle_heading(nested_heading, prefix, headings)
            self._walk_container(
                node,
                prefix,
                headings,
                excluded_nodes=frozenset({id(nested_heading)}),
            )
            return

        title = clean_text(node.get("name") or node.get("title") or "")
        if title:
            while headings and headings[-1][0] >= 2:
                headings.pop()
            headings.append((2, title))
            self._emit("heading", title, self._path(prefix, headings))
        self._walk_container(node, prefix, headings)

    def _handle_chunk(
        self,
        node: Tag,
        prefix: tuple[str, ...],
        headings: list[tuple[int, str]],
    ) -> None:
        title = clean_text(node.get("title", ""))
        if not title:
            self._walk_container(node, prefix, headings)
            return
        scoped_path = self._path(prefix, headings) + (title,)
        self._emit("chunk_title", title, scoped_path)
        self._walk_container(node, prefix=scoped_path, headings=[])

    def _handle_scoped_title(
        self,
        block_type: str,
        node: Tag,
        prefix: tuple[str, ...],
        headings: list[tuple[int, str]],
    ) -> tuple[str, ...]:
        title = self._node_text(node)
        current_path = self._path(prefix, headings)
        if not normalize_text(title):
            return current_path
        scoped_path = current_path + (title,)
        self._emit(block_type, title, scoped_path, self._node_links(node))
        return scoped_path

    def _handle_spoiler(
        self,
        node: Tag,
        prefix: tuple[str, ...],
        headings: list[tuple[int, str]],
    ) -> None:
        title_node = self._first_owned_by_class(node, "spoiler-text", "spoiler")
        content_node = self._first_owned_by_class(node, "spoiler-content", "spoiler")
        scoped_path = self._path(prefix, headings)
        if title_node is not None:
            scoped_path = self._handle_scoped_title(
                "spoiler_title", title_node, prefix, headings
            )
        if content_node is not None:
            self._walk_container(content_node, prefix=scoped_path, headings=[])
            return

        excluded = frozenset({id(title_node)}) if title_node is not None else frozenset()
        self._walk_container(node, prefix, headings, excluded_nodes=excluded)

    def _handle_tabset(
        self,
        node: Tag,
        prefix: tuple[str, ...],
        headings: list[tuple[int, str]],
    ) -> None:
        labels = self._owned_by_class(node, "tab-label", "tabset")
        panels = self._owned_by_class(node, "tab-panel", "tabset")
        current_path = self._path(prefix, headings)

        for index, label in enumerate(labels):
            title = self._node_text(label)
            scoped_path = current_path
            if normalize_text(title):
                scoped_path = current_path + (title,)
                self._emit(
                    "tab_title",
                    title,
                    scoped_path,
                    self._node_links(label),
                )
            if index < len(panels):
                self._walk_container(panels[index], prefix=scoped_path, headings=[])

        for panel in panels[len(labels) :]:
            self._walk_container(panel, prefix=current_path, headings=[])

    def _handle_factoid(
        self,
        node: Tag,
        prefix: tuple[str, ...],
        headings: list[tuple[int, str]],
    ) -> None:
        classes = set(node.get("class", ()))
        if "factoid_important" in classes:
            block_type = "factoid_important"
        elif "factoid_warning" in classes:
            block_type = "factoid_warning"
        else:
            block_type = "factoid_disclaimer"
        self._emit(
            block_type,
            self._node_text(node),
            self._path(prefix, headings),
            self._node_links(node),
        )

    def _handle_list_item(
        self,
        node: Tag,
        prefix: tuple[str, ...],
        headings: list[tuple[int, str]],
    ) -> None:
        excluded = frozenset({"ol", "ul"})
        self._emit(
            "list_item",
            self._node_text(node, excluded_tags=excluded),
            self._path(prefix, headings),
            self._node_links(node, excluded_tags=excluded),
        )
        nested_lists = [
            nested
            for nested in node.find_all(["ol", "ul"])
            if nested.find_parent("li") is node
        ]
        for nested_list in nested_lists:
            self._walk_container(nested_list, prefix, headings)

    def _handle_table(
        self,
        node: Tag,
        prefix: tuple[str, ...],
        headings: list[tuple[int, str]],
    ) -> None:
        active_headers: list[str] = []
        rows = [row for row in node.find_all("tr") if row.find_parent("table") is node]
        for row in rows:
            cells = [
                cell
                for cell in row.find_all(["th", "td"], recursive=False)
                if cell.find_parent("tr") is row
            ]
            if not cells:
                continue
            values = [self._node_text(cell) for cell in cells]
            links = self._node_links(row)
            if all(cell.name == "th" for cell in cells):
                active_headers = values
                self._emit(
                    "table_header",
                    " | ".join(values),
                    self._path(prefix, headings),
                    links,
                )
                continue

            if active_headers and len(active_headers) == len(values):
                parts = []
                for header, value in zip(active_headers, values):
                    if header and value:
                        parts.append(f"{header}: {value}")
                    elif header:
                        parts.append(f"{header}:")
                    else:
                        parts.append(value)
                text = " | ".join(parts)
            else:
                text = " | ".join(values)
            self._emit(
                "table_row",
                text,
                self._path(prefix, headings),
                links,
            )

    def _node_text(
        self,
        node: Tag,
        excluded_tags: AbstractSet[str] = frozenset(),
    ) -> str:
        parts: list[str] = []

        def collect(current: Tag | NavigableString) -> None:
            if isinstance(current, NavigableString):
                parts.append(str(current))
                return
            if current.name in REMOVED_NAMES or current.name in excluded_tags:
                return
            if current.name == "img":
                parts.append(str(current.get("alt", "")))
                return
            if current.name == "br":
                parts.append(" ")
                return
            for child in current.children:
                if isinstance(child, (Tag, NavigableString)):
                    collect(child)

        collect(node)
        return clean_text(" ".join(parts))

    def _node_links(
        self,
        node: Tag,
        excluded_tags: AbstractSet[str] = frozenset(),
    ) -> set[int]:
        result: set[int] = set()
        anchors = list(node.find_all("a", href=True))
        if node.name == "a" and node.has_attr("href"):
            anchors.insert(0, node)
        for anchor in anchors:
            ancestor = anchor.parent
            excluded = False
            while ancestor is not None and ancestor is not node:
                if isinstance(ancestor, Tag) and ancestor.name in excluded_tags:
                    excluded = True
                    break
                ancestor = ancestor.parent
            if excluded:
                continue
            article_id = self._article_id_from_href(str(anchor["href"]))
            if article_id is not None:
                result.add(article_id)
        return result

    def _article_id_from_href(self, href: str) -> int | None:
        parsed = urlparse(href)
        if parsed.netloc and parsed.netloc.casefold() != "support.avito.ru":
            return None
        match = ARTICLE_PATH_RE.search(parsed.path)
        if match is None:
            return None
        article_id = int(match.group(1))
        if article_id == self.article_id or article_id not in self.valid_article_ids:
            return None
        return article_id

    @staticmethod
    def _nearest_with_class(node: Tag, class_name: str) -> Tag | None:
        for ancestor in node.parents:
            if isinstance(ancestor, Tag) and class_name in ancestor.get("class", ()):
                return ancestor
        return None

    def _owned_by_class(
        self,
        node: Tag,
        target_class: str,
        owner_class: str,
    ) -> list[Tag]:
        return [
            candidate
            for candidate in node.find_all(class_=target_class)
            if self._nearest_with_class(candidate, owner_class) is node
        ]

    def _first_owned_by_class(
        self,
        node: Tag,
        target_class: str,
        owner_class: str,
    ) -> Tag | None:
        candidates = self._owned_by_class(node, target_class, owner_class)
        return candidates[0] if candidates else None

    @staticmethod
    def _deduplicate(blocks: list[_DraftBlock]) -> list[_DraftBlock]:
        result: list[_DraftBlock] = []
        for block in blocks:
            if (
                result
                and result[-1].normalized_text == block.normalized_text
                and result[-1].section_path == block.section_path
            ):
                result[-1].outgoing_article_ids.update(block.outgoing_article_ids)
                continue
            result.append(block)
        return result


def preprocess_html(
    article_id: int,
    title: str,
    body_html: str,
    valid_article_ids: AbstractSet[int],
) -> tuple[ProcessedArticle, list[TextBlock]]:
    cleaned_title = clean_text(title)
    extractor = _Extractor(int(article_id), valid_article_ids)
    blocks = extractor.extract("" if body_html is None else str(body_html))
    outgoing_article_ids = tuple(
        sorted({link for block in blocks for link in block.outgoing_article_ids})
    )
    article = ProcessedArticle(
        article_id=int(article_id),
        title=cleaned_title,
        clean_text="\n".join(block.clean_text for block in blocks),
        normalized_text="\n".join(block.normalized_text for block in blocks),
        outgoing_article_ids=outgoing_article_ids,
    )
    return article, blocks
