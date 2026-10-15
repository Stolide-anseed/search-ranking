from __future__ import annotations

from preprocessing import normalize_text, preprocess_html


VALID_IDS = frozenset({1, 2, 3, 4})


def process(body: str):
    return preprocess_html(1, "  Test\xa0title  ", body, VALID_IDS)


def test_normalize_text() -> None:
    assert normalize_text("  Ёлка, <b>НЕ HTML</b>!  ") == "елка b не html b"


def test_headings_update_section_path() -> None:
    _, blocks = process(
        "<h2>Main</h2><p>One</p><h3>Sub</h3><p>Two</p>"
        "<h2>Next</h2><p>Three</p>"
    )

    assert [(block.block_type, block.section_path) for block in blocks] == [
        ("heading", ("Main",)),
        ("paragraph", ("Main",)),
        ("heading", ("Main", "Sub")),
        ("paragraph", ("Main", "Sub")),
        ("heading", ("Next",)),
        ("paragraph", ("Next",)),
    ]


def test_headline_does_not_duplicate_nested_heading() -> None:
    _, blocks = process(
        '<headline name="Ignored"><h2>Actual heading</h2></headline><p>Body</p>'
    )

    assert [block.clean_text for block in blocks] == ["Actual heading", "Body"]
    assert [block.block_type for block in blocks] == ["heading", "paragraph"]


def test_chunk_scope_is_restored_after_chunk() -> None:
    _, blocks = process(
        '<h2>Outer</h2><chunk title="Scoped"><p>Inside</p></chunk><p>After</p>'
    )

    assert [(block.clean_text, block.section_path) for block in blocks] == [
        ("Outer", ("Outer",)),
        ("Scoped", ("Outer", "Scoped")),
        ("Inside", ("Outer", "Scoped")),
        ("After", ("Outer",)),
    ]


def test_spoiler_and_tab_titles_scope_their_content() -> None:
    body = """
    <h2>Root</h2>
    <div class="spoiler">
      <div class="spoiler-text"><p>Details</p></div>
      <div class="spoiler-content"><p>Spoiler body</p></div>
    </div>
    <div class="tabset">
      <label class="tab-label">Buyer</label>
      <label class="tab-label">Seller</label>
      <div class="tab-panels">
        <div class="tab-panel"><p>Buy</p></div>
        <div class="tab-panel"><p>Sell</p></div>
      </div>
    </div>
    """
    _, blocks = process(body)

    assert [(block.block_type, block.clean_text, block.section_path) for block in blocks] == [
        ("heading", "Root", ("Root",)),
        ("spoiler_title", "Details", ("Root", "Details")),
        ("paragraph", "Spoiler body", ("Root", "Details")),
        ("tab_title", "Buyer", ("Root", "Buyer")),
        ("paragraph", "Buy", ("Root", "Buyer")),
        ("tab_title", "Seller", ("Root", "Seller")),
        ("paragraph", "Sell", ("Root", "Seller")),
    ]


def test_factoid_is_one_typed_block() -> None:
    _, blocks = process(
        '<div class="factoid factoid_warning"><p>First</p><p>Second</p></div>'
    )

    assert len(blocks) == 1
    assert blocks[0].block_type == "factoid_warning"
    assert blocks[0].clean_text == "First Second"


def test_nested_lists_create_one_block_per_item() -> None:
    _, blocks = process(
        "<ul><li><p>Parent</p><ul><li>Child</li></ul></li><li>Sibling</li></ul>"
    )

    assert [block.block_type for block in blocks] == [
        "list_item",
        "list_item",
        "list_item",
    ]
    assert [block.clean_text for block in blocks] == ["Parent", "Child", "Sibling"]


def test_table_headers_are_applied_and_nested_table_is_not_repeated() -> None:
    body = """
    <table>
      <tr><th>Name</th><th>Value</th></tr>
      <tr><td>First</td><td><table><tr><td>Nested</td></tr></table></td></tr>
      <tr><td>Only one cell</td></tr>
    </table>
    """
    _, blocks = process(body)

    assert [(block.block_type, block.clean_text) for block in blocks] == [
        ("table_header", "Name | Value"),
        ("table_row", "Name: First | Value: Nested"),
        ("table_row", "Only one cell"),
    ]


def test_images_use_alt_text_without_duplicate_blocks() -> None:
    _, blocks = process(
        '<p>Open <img alt="Settings screen" src="x.png"></p>'
        '<img alt="Standalone image" src="y.png">'
    )

    assert [(block.block_type, block.clean_text) for block in blocks] == [
        ("paragraph", "Open Settings screen"),
        ("image_alt", "Standalone image"),
    ]


def test_links_are_filtered_and_self_link_is_removed() -> None:
    body = """
    <p>
      <a href="https://support.avito.ru/articles/2?x=1">Valid</a>
      <a href="/articles/3#part">Relative</a>
      <a href="/articles/1">Self</a>
      <a href="/articles/999">Unknown</a>
      <a href="https://example.com/articles/4">External</a>
    </p>
    """
    article, blocks = process(body)

    assert blocks[0].outgoing_article_ids == (2, 3)
    assert article.outgoing_article_ids == (2, 3)


def test_consecutive_duplicates_merge_links_but_global_duplicates_remain() -> None:
    body = """
    <h2>First</h2>
    <p><a href="/articles/2">Repeated</a></p>
    <p><a href="/articles/3">Repeated</a></p>
    <h2>Second</h2>
    <p>Repeated</p>
    """
    _, blocks = process(body)

    repeated = [block for block in blocks if block.clean_text == "Repeated"]
    assert len(repeated) == 2
    assert repeated[0].section_path == ("First",)
    assert repeated[0].outgoing_article_ids == (2, 3)
    assert repeated[1].section_path == ("Second",)


def test_block_ids_are_assigned_after_deduplication() -> None:
    article, blocks = process("<p>Same</p><p>Same</p><p>Different</p>")

    assert article.title == "Test title"
    assert [block.block_id for block in blocks] == ["1:0000", "1:0001"]
    assert [block.position for block in blocks] == [0, 1]
    assert article.clean_text == "Same\nDifferent"
    assert article.normalized_text == "same\ndifferent"


def test_unknown_containers_and_direct_text_are_preserved() -> None:
    _, blocks = process("<div>Before <custom><span>inside</span></custom> after</div>")

    assert [block.clean_text for block in blocks] == ["Before", "inside", "after"]


def test_direct_inline_link_is_attached_to_text_block() -> None:
    _, blocks = process('<div><a href="/articles/2">Direct link</a></div>')

    assert blocks[0].block_type == "text"
    assert blocks[0].outgoing_article_ids == (2,)
