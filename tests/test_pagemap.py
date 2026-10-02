"""Locating DOCX blocks in a rendered PDF: text pages, forward-only cursor, images, split tables."""

import pymupdf
import pytest

from ae.extract.pagemap import PageLocator, tables_by_page
from ae.schema import BBox

FALLBACK = (1, BBox(x0=10, y0=10, x1=290, y1=390))


@pytest.fixture
def render(tmp_path):
    """Two 300x400 pages: three text lines each (one line repeated on both) and one image per page."""
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), 0)
    pix.set_rect(pix.irect, (10, 10, 10))
    pages = [
        (["Introduction to the battery pack", "Row alpha nominal voltage 400 V"], pymupdf.Rect(20, 100, 120, 200)),
        (["Thermal management section", "Row beta maximum current 320 A"], pymupdf.Rect(20, 100, 220, 300)),
    ]
    for lines, img in pages:
        page = doc.new_page(width=300, height=400)
        for i, line in enumerate([*lines, "Shared phrase appears here"]):
            page.insert_text((20, 40 + 20 * i), line, fontsize=10)
        page.insert_image(img, pixmap=pix)
    path = tmp_path / "render.pdf"
    doc.save(path)
    return path


def test_page_sizes(render):
    assert PageLocator(render).sizes == [(300, 400), (300, 400)]


def test_locate_text_finds_the_page_and_box(render):
    loc = PageLocator(render)
    page, box = loc.locate_text("Thermal management section")
    assert page == 2 and box.x0 == pytest.approx(20) and box.y1 < 50


def test_locate_text_falls_back_to_a_shorter_prefix(render):
    hit = PageLocator(render).locate_text("Introduction to the battery pack and then words that are not rendered")
    assert hit is not None and hit[0] == 1


def test_locate_text_cursor_only_moves_forward(render):
    loc = PageLocator(render)
    assert loc.locate_text("Shared phrase appears here")[0] == 1
    assert loc.locate_text("Thermal management section")[0] == 2
    assert loc.locate_text("Shared phrase appears here")[0] == 2  # the page-1 copy is behind the cursor
    assert loc.locate_text("Introduction to the battery pack") is None


def test_locate_text_missing_or_too_short(render):
    loc = PageLocator(render)
    assert loc.locate_text("zzz not present anywhere") is None
    assert loc.locate_text("ab") is None


def test_next_image_walks_images_in_document_order(render):
    loc = PageLocator(render)
    first, second = loc.next_image(), loc.next_image()
    assert first[0] == 1 and (first[1].x0, first[1].y0, first[1].x1, first[1].y1) == (20, 100, 120, 200)
    assert second[0] == 2 and second[1].x1 == 220
    assert loc.next_image() is None


def test_tables_by_page_splits_rows_across_pages(render):
    rows = [
        ["#", "Description"],
        ["1", "Row alpha nominal voltage 400 V"],
        ["2", "short"],  # no distinctive cell: inherits the previous row's page
        ["3", "Row beta maximum current 320 A"],
    ]
    pieces = tables_by_page(rows, PageLocator(render), "Specs", "t1", FALLBACK)
    assert [(p, t.rows, t.continued) for p, t in pieces] == [
        (1, [rows[0], rows[1], rows[2]], False),
        (2, [rows[0], rows[3]], True),
    ]
    assert all(t.table_id == "t1" and t.title == "Specs" for _, t in pieces)
    assert all(t.bbox.x0 == 10 and t.bbox.x1 == 290 for _, t in pieces)  # x from the fallback, y from the hits


def test_tables_by_page_without_renderer_keeps_one_piece_on_fallback_page():
    rows = [["h"], ["r1"], ["r2"]]
    ((page, table),) = tables_by_page(rows, None, "Specs", "t1", FALLBACK)
    assert page == 1 and table.rows == rows and table.bbox == FALLBACK[1] and not table.continued


def test_tables_by_page_empty_table():
    assert tables_by_page([], None, "Specs", "t1", FALLBACK) == []
