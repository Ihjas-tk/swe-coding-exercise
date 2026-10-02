"""Reading order, paragraph merging and role tagging for native text blocks."""

import random

from ae.extract.native.text import (
    RawBlock,
    build_text_blocks,
    classify_role,
    detect_columns,
    merge_adjacent,
    order_blocks,
)
from ae.schema import BBox

PAGE_H = 792


def rb(x0, y0, x1, y1, text, size=10.0, bold=False) -> RawBlock:
    return RawBlock(BBox(x0=x0, y0=y0, x1=x1, y1=y1), text, size, bold)


def two_column_page() -> list[RawBlock]:
    """Letter page: full-width title, then two 224 pt columns with a 20 pt gutter."""
    blocks = [rb(72, 40, 540, 70, "TITLE", 16, True)]
    for i, y in enumerate((100, 200, 300), start=1):
        blocks.append(rb(72, y, 296, y + 80, f"L{i}"))
        blocks.append(rb(316, y, 540, y + 80, f"R{i}"))
    return blocks


def test_two_columns_are_detected_from_the_gutter():
    assert detect_columns(two_column_page(), 72, 540) == [(72, 296), (316, 540)]


def test_title_then_left_column_then_right_column():
    blocks = two_column_page()
    random.Random(0).shuffle(blocks)  # content-stream order must not matter
    ordered = order_blocks(blocks, PAGE_H)
    assert [b.text for b, _ in ordered] == ["TITLE", "L1", "L2", "L3", "R1", "R2", "R3"]
    assert [c for _, c in ordered] == [None, 0, 0, 0, 1, 1, 1]


def test_full_width_banner_starts_a_new_band():
    blocks = [
        *two_column_page(),
        rb(72, 400, 540, 420, "BANNER", 12, True),
        rb(72, 430, 296, 500, "L4"),
        rb(316, 430, 540, 500, "R4"),
    ]
    assert [b.text for b, _ in order_blocks(blocks, PAGE_H)] == [
        "TITLE", "L1", "L2", "L3", "R1", "R2", "R3", "BANNER", "L4", "R4",
    ]  # fmt: skip


def test_single_column_page_is_top_to_bottom():
    blocks = [rb(72, 300, 540, 340, "B"), rb(72, 100, 540, 140, "A"), rb(72, 500, 540, 540, "C")]
    assert [(b.text, c) for b, c in order_blocks(blocks, PAGE_H)] == [("A", 0), ("B", 0), ("C", 0)]


def test_order_blocks_on_empty_input():
    assert order_blocks([], PAGE_H) == []


def test_merge_adjacent_joins_centred_caption_continuation():
    caption = [
        rb(72, 500, 540, 512, "FIG. 2 — Perspective cross section of the gripper showing the control unit and", 9),
        rb(200, 513, 412, 525, "the electromagnet assembly.", 9),
    ]
    merged = merge_adjacent(caption)
    assert len(merged) == 1
    assert merged[0].text.endswith("control unit and the electromagnet assembly.")
    assert (merged[0].bbox.x0, merged[0].bbox.y0, merged[0].bbox.x1, merged[0].bbox.y1) == (72, 500, 540, 525)


def test_merge_adjacent_dehyphenates_split_paragraph():
    merged = merge_adjacent(
        [rb(72, 100, 300, 150, "First part of a para-"), rb(72, 151, 300, 200, "graph continues here.")]
    )
    assert [b.text for b in merged] == ["First part of a paragraph continues here."]


def test_merge_adjacent_keeps_separate_paragraphs_captions_and_size_changes():
    body = rb(72, 100, 300, 150, "Body text here.")
    assert len(merge_adjacent([body, rb(72, 170, 300, 190, "Next paragraph.")])) == 2  # gap too large
    assert len(merge_adjacent([body, rb(72, 151, 300, 170, "FIG. 3 — A new caption")])) == 2
    assert len(merge_adjacent([body, rb(72, 151, 300, 170, "Footnote text", 7)])) == 2


def test_merge_adjacent_does_not_mutate_input_blocks():
    a, b = rb(72, 100, 300, 150, "one"), rb(72, 151, 300, 200, "two")
    merge_adjacent([a, b])
    assert a.text == "one" and b.text == "two"


def test_classify_role_heading_by_numbered_bold_text_and_by_size():
    assert classify_role(rb(72, 300, 540, 320, "3. Electrical Specifications", 10, True), 10, PAGE_H) == "heading"
    assert classify_role(rb(72, 300, 540, 330, "ROBOTIC GRIPPER", 16), 10, PAGE_H) == "heading"


def test_classify_role_caption():
    assert classify_role(rb(72, 300, 540, 320, "FIG. 2 — cross section", 9), 10, PAGE_H) == "caption"
    assert classify_role(rb(72, 300, 540, 320, "Table 3: Pin assignment", 9), 10, PAGE_H) == "caption"


def test_classify_role_running_header_and_footer():
    assert (
        classify_role(rb(72, 20, 540, 40, "US 8,485,576 B2   Jul. 16, 2013   Sheet 1 of 3", 9), 10, PAGE_H) == "header"
    )
    assert classify_role(rb(72, 760, 540, 780, "Page 3", 9), 10, PAGE_H) == "footer"


def test_classify_role_large_bold_text_in_header_band_is_a_heading_not_header():
    assert classify_role(rb(72, 20, 540, 40, "BIG TITLE", 18, True), 10, PAGE_H) == "heading"


def test_classify_role_body():
    text = "The quick body text of a paragraph that goes on for a while and so on."
    assert classify_role(rb(72, 300, 540, 400, text), 10, PAGE_H) == "body"


def test_build_text_blocks_tags_roles_and_columns_in_reading_order():
    out = build_text_blocks(two_column_page(), PAGE_H)
    assert [(b.text, b.role, b.column) for b in out][:3] == [
        ("TITLE", "heading", None),
        ("L1", "body", 0),
        ("L2", "body", 0),
    ]
    assert all(b.source == "text_layer" for b in out)
