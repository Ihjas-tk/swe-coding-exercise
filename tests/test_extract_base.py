"""The extraction entry point: layout merge on OCR'd pages, file-type check, index paths."""

from pathlib import Path

import pytest

from ae.extract.base import _merge_layout, parse
from ae.schema import BBox, FigureBlock, Page, TableBlock, TextBlock


def box(x0, y0, x1, y1):
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def scanned_page() -> Page:
    return Page(
        number=1,
        width=612,
        height=792,
        is_scanned=True,
        blocks=[
            TextBlock(bbox=box(72, 72, 540, 100), text="Heading text", source="ocr"),
            TextBlock(bbox=box(80, 210, 300, 220), text="cell text OCR'd as prose", source="ocr"),
            FigureBlock(bbox=box(72, 400, 540, 700), caption="FIG. 1 — drawing", figure_id="FIG. 1"),
        ],
    )


def test_merge_adds_tables_and_drops_text_inside_them():
    page = scanned_page()
    table = TableBlock(bbox=box(72, 200, 540, 300), rows=[["a", "b"], ["1", "2"]])
    _merge_layout(page, Page(number=1, width=612, height=792, blocks=[table]))
    texts = [b.text for b in page.blocks if isinstance(b, TextBlock)]
    assert texts == ["Heading text"]
    assert [b.kind for b in page.blocks] == ["text", "table", "figure"]  # re-ordered top to bottom


def test_merge_skips_empty_tables_and_duplicate_figures():
    page = scanned_page()
    layout = Page(
        number=1,
        width=612,
        height=792,
        blocks=[
            TableBlock(bbox=box(72, 200, 540, 300), rows=[]),  # empty grid: ignored, text kept
            FigureBlock(bbox=box(75, 405, 535, 690)),  # same figure as native's: native's kept
            FigureBlock(bbox=box(72, 110, 300, 190)),  # new figure: added
        ],
    )
    _merge_layout(page, layout)
    figs = [b for b in page.blocks if isinstance(b, FigureBlock)]
    assert len(figs) == 2 and figs[1].caption == "FIG. 1 — drawing"
    assert len([b for b in page.blocks if isinstance(b, TextBlock)]) == 2


def test_parse_rejects_unsupported_files(tmp_path: Path):
    f = tmp_path / "notes.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="cannot extract"):
        parse(f)


def test_index_db_namespaces(monkeypatch):
    from ae import config
    from ae.index.store import index_db

    monkeypatch.setattr(config, "INDEX", "")
    assert index_db() == Path("data/index/default/index.sqlite")
    assert index_db("ext") == Path("data/index/ext/index.sqlite")
    monkeypatch.setattr(config, "INDEX", "smoke")
    assert index_db() == Path("data/index/smoke/index.sqlite")
    assert index_db("") == Path("data/index/default/index.sqlite")
