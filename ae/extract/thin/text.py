"""Text blocks and reading order (thin backend).

How PyMuPDF gives us text
-------------------------
`page.get_text("dict")` walks the page's content stream and returns *blocks* ->
*lines* -> *spans*. A block is a run of lines MuPDF grouped by proximity and font;
each level carries a bbox in PDF points with a top-left origin. MuPDF emits blocks
in content-stream order (the order the PDF writer drew them), which is usually but
not reliably the reading order, so we never trust it: we re-sort geometrically.

Reading order algorithm (ours, not MuPDF's)
-------------------------------------------
1. Measure the text area (min x0 .. max x1 over all blocks).
2. A block wider than FULL_WIDTH_FRAC of that area is a *span* block (title,
   section banner). Span blocks split the page into horizontal *bands*.
3. Inside the non-span blocks, find vertical gutters: merge the blocks'
   x-intervals; every gap wider than MIN_GUTTER is a column boundary. This gives
   N columns without assuming N == 2.
4. Order = for each band top to bottom: the span block, then column 0's blocks
   top to bottom, then column 1's, ...

Why this and not `sort=True`: MuPDF's sort flag orders by (y, x), which on a
two-column patent interleaves the columns line by line, exactly the failure the
brief warns about.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import pymupdf

from ae.schema import BBox, TextBlock

FULL_WIDTH_FRAC = 0.62  # block wider than this fraction of text area => spans columns
MIN_GUTTER = 12.0  # points; smaller x-gaps between blocks are not column gutters
MERGE_GAP = 3.0  # points; consecutive blocks closer than this (same size) are one paragraph
HEADER_FRAC = 0.12  # top/bottom fraction of page height treated as running header/footer (patent headers sit at ~10%)
CAPTION_RE = re.compile(r"^\s*(FIG(?:URE)?\.?|Figure|Table)\s*\d+[A-Za-z]?\s*([—–:\-]|\.?\s*$)", re.I)
FIG_TITLE_RE = re.compile(r"^\s*(FIG(?:URE)?\.?|Figure)\s*\d+[A-Za-z]?\.?\s*$", re.I)  # bare "FIG. 2" above a drawing
HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*\.?\s+\S|[A-Z][A-Z \-&/]{5,}$)")


@dataclass
class RawBlock:
    bbox: BBox
    text: str
    size: float  # dominant font size
    bold: bool
    source: str = "text_layer"
    conf: float | None = None


def raw_blocks_from_page(page: pymupdf.Page) -> list[RawBlock]:
    """Text blocks straight from the PDF text layer (no OCR)."""
    out: list[RawBlock] = []
    for b in page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        if b["type"] != 0:
            continue
        spans = [s for l in b["lines"] for s in l["spans"] if s["text"].strip()]
        if not spans:
            continue
        lines = [" ".join(s["text"] for s in l["spans"]).strip() for l in b["lines"]]
        text = _join_lines([l for l in lines if l])
        sizes = [s["size"] for s in spans]
        bold = sum(1 for s in spans if "bold" in s["font"].lower() or s["flags"] & 16) > len(spans) / 2
        x0, y0, x1, y1 = b["bbox"]
        out.append(RawBlock(BBox(x0=x0, y0=y0, x1=x1, y1=y1), text, statistics.median(sizes), bold))
    return out


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines; de-hyphenate 'exam-\\nple' -> 'example'."""
    text = ""
    for l in lines:
        if text.endswith("-") and l[:1].islower():
            text = text[:-1] + l
        else:
            text = (text + " " + l) if text else l
    return re.sub(r"[ \t]+", " ", text).strip()


def merge_adjacent(blocks: list[RawBlock]) -> list[RawBlock]:
    """Merge blocks MuPDF split mid-paragraph: same column, same size, tiny vertical gap.

    A centred last line (e.g. the second line of a caption) has a different x0 from
    its paragraph, so "same column" means horizontally contained, not x0-aligned.
    """
    blocks = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))
    out: list[RawBlock] = []
    for b in blocks:
        target = None
        for p in reversed(out[-6:]):
            contained = b.bbox.x0 >= p.bbox.x0 - 3 and b.bbox.x1 <= p.bbox.x1 + 3
            aligned = abs(p.bbox.x0 - b.bbox.x0) < 3
            gap = b.bbox.y0 - p.bbox.y1
            if (contained or aligned) and -2 <= gap < MERGE_GAP and abs(p.size - b.size) < 0.6 and not CAPTION_RE.match(b.text):
                target = p
                break
        if target is not None:
            target.text = _join_lines([target.text, b.text])
            target.bbox = BBox(
                x0=min(target.bbox.x0, b.bbox.x0), y0=target.bbox.y0, x1=max(target.bbox.x1, b.bbox.x1), y1=max(target.bbox.y1, b.bbox.y1)
            )
            continue
        out.append(RawBlock(b.bbox, b.text, b.size, b.bold, b.source, b.conf))
    return out


def detect_columns(blocks: list[RawBlock], text_x0: float, text_x1: float) -> list[tuple[float, float]]:
    """Return column x-ranges from the union of narrow blocks' x-intervals."""
    width = text_x1 - text_x0
    narrow = [b for b in blocks if b.bbox.width < FULL_WIDTH_FRAC * width]
    if not narrow:
        return [(text_x0, text_x1)]
    ivs = sorted((b.bbox.x0, b.bbox.x1) for b in narrow)
    merged = [list(ivs[0])]
    for x0, x1 in ivs[1:]:
        if x0 <= merged[-1][1] + MIN_GUTTER:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    # Drop slivers (e.g. a lone page number) that would create bogus columns.
    cols = [(a, b) for a, b in merged if (b - a) > 0.15 * width] or [(text_x0, text_x1)]
    return cols


def order_blocks(blocks, page_w: float, page_h: float) -> list[tuple[object, int | None]]:
    """Return (item, column_index) in reading order for any items with a .bbox.

    column_index None == spans all columns. Used twice: on raw text blocks (to tag
    roles and columns) and again on the final mix of text/table/figure blocks so
    tables and figures land where a reader would meet them.
    """
    if not blocks:
        return []
    tx0 = min(b.bbox.x0 for b in blocks)
    tx1 = max(b.bbox.x1 for b in blocks)
    width = tx1 - tx0
    cols = detect_columns(blocks, tx0, tx1)

    def col_of(b: RawBlock) -> int | None:
        if b.bbox.width >= FULL_WIDTH_FRAC * width or len(cols) == 1:
            return None if len(cols) > 1 else 0
        cx = b.bbox.cx
        return min(range(len(cols)), key=lambda i: abs((cols[i][0] + cols[i][1]) / 2 - cx))

    tagged = [(b, col_of(b)) for b in blocks]
    spans = sorted([b for b, c in tagged if c is None], key=lambda b: b.bbox.y0)
    # Band boundaries: y0 of each span block.
    bounds = [-1.0] + [s.bbox.y0 for s in spans] + [page_h + 1]
    ordered: list[tuple[RawBlock, int | None]] = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if i > 0:
            ordered.append((spans[i - 1], None))
        in_band = [(b, c) for b, c in tagged if c is not None and lo <= b.bbox.y0 < hi]
        for ci in range(len(cols)):
            ordered += sorted([(b, c) for b, c in in_band if c == ci], key=lambda t: t[0].bbox.y0)
    return ordered


def classify_role(b: RawBlock, body_size: float, page_h: float) -> str:
    if CAPTION_RE.match(b.text) and len(b.text) < 400:
        return "caption"
    top, bottom = b.bbox.y0 < HEADER_FRAC * page_h, b.bbox.y1 > (1 - HEADER_FRAC) * page_h
    if (top or bottom) and len(b.text) < 80 and b.size <= body_size + 0.5:
        return "header" if top else "footer"
    if (b.bold and b.size >= body_size * 1.1) or (b.bold and len(b.text) < 60 and HEADING_RE.match(b.text)):
        return "heading"
    if b.size >= body_size * 1.3 and len(b.text) < 120:
        return "heading"
    return "body"


def build_text_blocks(raw: list[RawBlock], page_w: float, page_h: float) -> list[TextBlock]:
    raw = merge_adjacent(raw)
    if not raw:
        return []
    body_size = statistics.median([b.size for b in raw for _ in range(max(1, len(b.text) // 40))])
    out = []
    for b, col in order_blocks(raw, page_w, page_h):
        out.append(
            TextBlock(
                bbox=b.bbox,
                text=b.text,
                role=classify_role(b, body_size, page_h),
                column=col,
                source=b.source,  # type: ignore[arg-type]
                ocr_confidence=b.conf,
            )
        )
    return out
