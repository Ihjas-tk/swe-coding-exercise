"""Chunking of a synthetic ParsedDocument: kinds, prefixes, rows, figures, claims, sizes."""

import pytest

from ae.index.chunker import MAX_TOKENS, TARGET_TOKENS, chunk_document, n_tokens, row_text
from ae.index.numerals import NumeralIndex
from ae.schema import BBox, FigureBlock, Page, ParsedDocument, TableBlock, TextBlock


def box(y: float) -> BBox:
    return BBox(x0=72, y0=y, x1=540, y1=y + 20)


def text(t: str, role: str = "body", y: float = 100) -> TextBlock:
    return TextBlock(bbox=box(y), text=t, role=role)


def words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


HEADER = ["Parameter", "Value"]


def make_doc() -> ParsedDocument:
    p1 = Page(
        number=1,
        width=612,
        height=792,
        blocks=[
            text("Battery Controller", "heading", 40),
            text("1. Overview", "heading", 70),
            text("The control unit 160 regulates the electromagnet 120a. " * 3, y=100),
            TableBlock(
                bbox=box(300),
                rows=[HEADER, ["Max discharge current", "320 A"], ["Nominal voltage", "400 V"]],
                title="3. Electrical",
                table_id="t1",
            ),
            text(words(80, "tail"), y=500),
        ],
    )
    p2 = Page(
        number=2,
        width=612,
        height=792,
        blocks=[
            TableBlock(
                bbox=box(50), rows=[HEADER, ["Cell count", "96"]], title="3. Electrical", table_id="t1", continued=True
            ),
            text(words(80, "head"), y=90),
            text("2. Figures", "heading", 400),
            FigureBlock(
                bbox=box(450),
                caption="FIG. 2 — Block diagram of the controller",
                figure_id="FIG. 2",
                ocr_labels=["160", "20", "housing"],
            ),
            text("Claims", "heading", 600),
            text("1. A gripper comprising a control unit 160.", y=620),
            text("2. The gripper of claim 1, wherein the electromagnet 120a is energised.", y=650),
        ],
    )
    return ParsedDocument(doc="X.pdf", source_path="X.pdf", backend="native", pages=[p1, p2])


@pytest.fixture(scope="module")
def chunks():
    doc = make_doc()
    idx = NumeralIndex()
    idx.add_document(doc)
    return chunk_document(doc, idx)


def test_chunk_kinds_in_document_order(chunks):
    assert [c.kind for c in chunks] == [
        "text", "table_row", "table_row", "table", "text",
        "table_row", "table", "text", "figure", "claim", "claim",
    ]  # fmt: skip


def test_prefix_is_title_section_page(chunks):
    assert chunks[0].prefix == "Battery Controller › 1. Overview › p.1"
    fig = next(c for c in chunks if c.kind == "figure")
    assert fig.prefix == "Battery Controller › 2. Figures › p.2"
    assert chunks[0].full_text.startswith(chunks[0].prefix + "\n")


def test_chunk_ids_are_unique_and_page_scoped(chunks):
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[0].id == "X.pdf#p1#text1"


def test_table_rows_are_serialised_with_title_and_column_names(chunks):
    rows = [c for c in chunks if c.kind == "table_row"]
    assert rows[0].text == "Table: 3. Electrical | Parameter: Max discharge current | Value: 320 A"
    assert "320a" in rows[0].identifiers
    assert rows[0].meta["table_id"] == "t1" and rows[0].meta["row"] == 1


def test_split_table_rows_cite_their_own_page_and_share_table_id(chunks):
    rows = [c for c in chunks if c.kind == "table_row"]
    assert [(c.page, c.meta["continued"]) for c in rows] == [(1, False), (1, False), (2, True)]
    assert {c.meta["table_id"] for c in rows} == {"t1"}
    assert rows[2].text == "Table: 3. Electrical | Parameter: Cell count | Value: 96"


def test_whole_table_chunk_is_markdown_with_title(chunks):
    tables = [c for c in chunks if c.kind == "table"]
    assert tables[0].text.startswith("3. Electrical\n| Parameter | Value |")
    assert tables[0].meta["n_rows"] == 2 and tables[1].meta["n_rows"] == 1


def test_figure_chunk_has_caption_labels_and_glossary(chunks):
    fig = next(c for c in chunks if c.kind == "figure")
    assert fig.text.startswith("FIG. 2 — Block diagram of the controller")
    assert "Labels: 160 | 160 = control unit |" in fig.text
    assert "Also defined on this page: 120a = electromagnet" in fig.text
    assert "Text in figure: housing" in fig.text
    assert {"fig2", "ref160"} <= set(fig.identifiers)
    assert fig.meta["labels"] == {"matched": ["160"], "repaired": [], "unmatched": ["20"]}


def test_figure_label_matched_and_repaired_to_same_numeral_is_listed_once():
    doc = make_doc()
    fig = next(b for b in doc.pages[1].blocks if isinstance(b, FigureBlock))
    fig.ocr_labels = ["160", "60"]
    idx = NumeralIndex()
    idx.add_document(doc)
    chunk = next(c for c in chunk_document(doc, idx) if c.kind == "figure")
    assert "Labels: 160 | 160 = control unit |" in chunk.text


def test_claims_are_one_chunk_each_with_claim_number(chunks):
    claims = [c for c in chunks if c.kind == "claim"]
    assert [c.meta["claim"] for c in claims] == [1, 2]
    assert all(c.section == "Claims" for c in claims)


def test_text_chunks_never_cross_a_page(chunks):
    texts = [c for c in chunks if c.kind == "text"]
    assert any("tail0" in c.text for c in texts if c.page == 1)
    assert all("head0" not in c.text for c in texts if c.page == 1)
    assert all("tail0" not in c.text for c in texts if c.page == 2)


def test_headings_become_sections_not_chunks(chunks):
    assert not any(c.text in ("Battery Controller", "1. Overview", "2. Figures", "Claims") for c in chunks)


def test_body_blocks_pack_within_token_bounds():
    blocks = [text(words(100, f"b{i}_"), y=50 + i * 10) for i in range(12)]
    doc = ParsedDocument(
        doc="Y.pdf", source_path="Y.pdf", backend="native", pages=[Page(number=1, width=612, height=792, blocks=blocks)]
    )
    texts = [c for c in chunk_document(doc) if c.kind == "text"]
    assert len(texts) > 1
    assert all(n_tokens(c.text) <= MAX_TOKENS for c in texts)
    assert all(n_tokens(c.text) >= TARGET_TOKENS * 0.4 for c in texts[:-1])
    assert sum(len(c.text.split()) for c in texts) == 1200  # nothing lost or duplicated


def test_small_trailing_text_is_merged_into_predecessor():
    blocks = [text(words(150, "a"), y=100), text("1. Next", "heading", 200)]
    blocks2 = [text(words(200, "big"), y=100), text(words(250, "more"), y=300), text("short tail", y=500)]
    doc = ParsedDocument(
        doc="Z.pdf",
        source_path="Z.pdf",
        backend="native",
        pages=[
            Page(number=1, width=612, height=792, blocks=blocks),
            Page(number=2, width=612, height=792, blocks=blocks2),
        ],
    )
    p2 = [c for c in chunk_document(doc) if c.page == 2]
    assert len(p2) == 2 and p2[-1].text.endswith("short tail")


def test_single_oversized_block_is_split_to_the_cap():
    doc = ParsedDocument(
        doc="W.pdf",
        source_path="W.pdf",
        backend="native",
        pages=[Page(number=1, width=612, height=792, blocks=[text(words(500, "w"))])],
    )
    assert all(n_tokens(c.text) <= MAX_TOKENS for c in chunk_document(doc))


def test_big_table_whole_chunk_falls_back_to_header_summary():
    rows = [["Name", "Description"]] + [[f"item{i}", words(10, f"d{i}_")] for i in range(40)]
    doc = ParsedDocument(
        doc="T.pdf",
        source_path="T.pdf",
        backend="native",
        pages=[Page(number=1, width=612, height=792, blocks=[TableBlock(bbox=box(100), rows=rows, title="Parts")])],
    )
    table = next(c for c in chunk_document(doc) if c.kind == "table")
    assert table.text == "Parts\nColumns: Name | Description\n(40 rows)"


def test_row_text_without_header_or_title_uses_column_numbers():
    t = TableBlock(bbox=box(0), rows=[["a", "", "c"]], has_header=False)
    assert row_text(t, ["a", "", "c"]) == "Table | col1: a | col3: c"


def test_doc_title_falls_back_to_file_stem():
    doc = ParsedDocument(
        doc="EV-BMS-100_design_document.pdf",
        source_path="x",
        backend="native",
        pages=[Page(number=1, width=612, height=792, blocks=[text(words(70, "x"))])],
    )
    assert chunk_document(doc)[0].prefix == "EV-BMS-100_design_document › p.1"
