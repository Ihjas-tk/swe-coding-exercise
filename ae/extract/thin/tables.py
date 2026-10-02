"""Table extraction (thin backend): pdfplumber for the grid, PyMuPDF for the text.

How pdfplumber finds tables
---------------------------
pdfplumber reads the page's vector graphics (`lines`, `rects`, `curves`) and
characters via pdfminer.six. `find_tables` builds a grid in four steps:
1. Edges: with the *lines* strategy, every horizontal/vertical line or rect edge
   becomes a candidate edge; with the *text* strategy, edges are synthesised from
   the left/right x of aligned words and the top/bottom y of text lines.
2. Edges are snapped together (`snap_tolerance`) and joined into longer segments
   (`join_tolerance`).
3. Intersections of horizontal and vertical edges become cell corners; the
   smallest rectangles bounded by intersections become cells.
4. Cells that share edges are grouped into tables.

Why the cell text comes from PyMuPDF, not pdfplumber
----------------------------------------------------
pdfminer decodes glyphs through the font's declared encoding and, on this corpus,
turns "≥" into "‡" and "•" into "(cid:127)"; MuPDF resolves the same glyphs
correctly. pdfplumber's `extract()` also assigns *characters* to cells, so when a
value physically overflows its cell (Falcon-VT1 "Processor" row, where the PDF
overprints "MHz" and "with" at the same x) the two cells' characters interleave
("MwiHthz"). We therefore take the grid from pdfplumber and fill each cell with
PyMuPDF *words* whose centre lies inside it, ordered by line then x. Overflow
still lands in the neighbouring cell, but words stay whole.

Policy for this corpus
----------------------
The design-document spec tables are drawn with ruling lines, so the *lines*
strategy is exact. The *text* strategy over-detects on prose (it turns a
two-column patent page into a 19x8 "table"), so it is used only when no ruled
table exists and only if its column edges never cut through a word
(`_word_integrity`), which prose virtually never satisfies.
"""
from __future__ import annotations

import re

import pdfplumber
import pymupdf

from ae.schema import BBox, TableBlock, TextBlock

LINES = {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 3, "join_tolerance": 3}
TEXT = {"vertical_strategy": "text", "horizontal_strategy": "text", "snap_tolerance": 3, "min_words_vertical": 3}

Word = tuple[float, float, float, float, str, int, int, int]  # PyMuPDF (x0,y0,x1,y1,text,block,line,word)


def _fill_cells(table: pdfplumber.table.Table, words: list[Word]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.rows:
        out_row = []
        for cell in row.cells:
            if cell is None:
                out_row.append("")
                continue
            x0, y0, x1, y1 = cell
            inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
            out_row.append(re.sub(r"\s+", " ", " ".join(w[4] for w in _by_line(inside))).strip())
        rows.append(out_row)
    return [r for r in rows if any(r)]


def _by_line(words: list[Word]) -> list[Word]:
    """Order words by visual line (vertical-centre clustering), then x. Symbols set in a
    different font (e.g. "≥") sit on a slightly different baseline; a fixed y-bin would
    push them onto their own line and after the number they precede."""
    words = sorted(words, key=lambda w: (w[1] + w[3]) / 2)
    lines: list[list[Word]] = []
    for w in words:
        cy, h = (w[1] + w[3]) / 2, (w[3] - w[1])
        if lines and abs(cy - sum((x[1] + x[3]) / 2 for x in lines[-1]) / len(lines[-1])) < 0.5 * h:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [w for line in lines for w in sorted(line, key=lambda w: w[0])]


def _merge_continuation_rows(rows: list[list[str]]) -> list[list[str]]:
    """A row whose first cell is empty is a wrapped continuation of the previous row."""
    out: list[list[str]] = []
    for r in rows:
        if out and r and r[0] == "" and any(r[1:]):
            prev = out[-1]
            for i, v in enumerate(r):
                if v and i < len(prev):
                    prev[i] = (prev[i] + " " + v).strip()
            continue
        out.append(list(r))
    return out


def _word_integrity(table: pdfplumber.table.Table, words: list[Word]) -> float:
    """Fraction of words inside the table bbox that are not cut by a column edge."""
    x0, y0, x1, y1 = table.bbox
    edges = sorted({c[0] for c in table.cells if c} | {c[2] for c in table.cells if c})
    inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    if not inside:
        return 0.0
    whole = sum(1 for w in inside if not any(w[0] + 1 < e < w[2] - 1 for e in edges))
    return whole / len(inside)


def _sane(rows: list[list[str]], bbox: BBox, page_area: float) -> bool:
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return False
    cells = [c for r in rows for c in r]
    if sum(1 for c in cells if c) / max(1, len(cells)) < 0.6:
        return False
    return bbox.width * bbox.height <= 0.7 * page_area


def extract_tables(pl_page: pdfplumber.page.Page, mu_page: pymupdf.Page, text_blocks: list[TextBlock]) -> list[TableBlock]:
    words: list[Word] = mu_page.get_text("words")
    page_area = float(pl_page.width * pl_page.height)
    found: list[TableBlock] = []
    for strategy, settings in (("lines", LINES), ("text", TEXT)):
        for t in pl_page.find_tables(table_settings=settings):
            x0, y0, x1, y1 = t.bbox
            bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
            if strategy == "text" and _word_integrity(t, words) < 0.98:
                continue
            rows = _merge_continuation_rows(_fill_cells(t, words))
            if not rows or (strategy == "text" and not _sane(rows, bbox, page_area)):
                continue
            found.append(TableBlock(bbox=bbox, rows=rows, has_header=True, title=_title_for(bbox, text_blocks)))
        if found:
            break  # ruled tables found; don't add text-strategy duplicates
    return found


def _title_for(bbox: BBox, text_blocks: list[TextBlock], max_gap: float = 80.0) -> str | None:
    """Nearest heading above the table (e.g. '3. Electrical Specifications')."""
    best, best_gap = None, max_gap
    for tb in text_blocks:
        if tb.role != "heading":
            continue
        gap = bbox.y0 - tb.bbox.y1
        if 0 <= gap < best_gap:
            best, best_gap = tb.text, gap
    return best


def drop_text_inside_tables(text_blocks: list[TextBlock], tables: list[TableBlock]) -> list[TextBlock]:
    """Text that lives inside a table bbox is already represented by the table's cells."""
    return [
        tb
        for tb in text_blocks
        if not any(t.bbox.x0 - 2 <= tb.bbox.cx <= t.bbox.x1 + 2 and t.bbox.y0 - 2 <= tb.bbox.cy <= t.bbox.y1 + 2 for t in tables)
    ]
