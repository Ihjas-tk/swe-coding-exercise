"""Text blocks and reading order (native backend).

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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import pymupdf

from ae.extract.captions import CAPTION_RE, FIG_TITLE_RE
from ae.schema import BBox, Role, TextBlock, TextSource

__all__ = [
    "CAPTION_RE",
    "FIG_TITLE_RE",
    "HasBBox",
    "RawBlock",
    "build_text_blocks",
    "classify_role",
    "detect_columns",
    "join_lines",
    "merge_adjacent",
    "order_blocks",
    "raw_blocks_from_page",
]

# Reading order --------------------------------------------------------------------------
FULL_WIDTH_FRAC = 0.62
"""A block wider than this fraction of the text area spans the columns (title, banner)."""
MIN_GUTTER = 12.0
"""Points; narrower x-gaps between blocks are word spacing, not a column gutter."""
MIN_COLUMN_FRAC = 0.15
"""A merged x-interval narrower than this fraction of the text area is a sliver (a lone page number), not a column."""

# Paragraph merging ----------------------------------------------------------------------
MERGE_GAP = 3.0
"""Points; consecutive same-size blocks closer than this vertically are one paragraph."""
MERGE_MAX_OVERLAP = 2.0
"""Points a following block may overlap its predecessor vertically and still merge."""
MERGE_X_TOLERANCE = 3.0
"""Points of slack for "horizontally contained" / "same x0" when merging."""
MERGE_SIZE_TOLERANCE = 0.6
"""Font-size difference (pt) above which two blocks are never one paragraph."""
MERGE_LOOKBACK = 6
"""How many recent blocks a block may merge into (MuPDF emits columns interleaved)."""
SPAN_SPACE_EM = 0.25
"""Horizontal gap between two spans, in em, above which a space is inserted."""
PYMUPDF_BOLD_FLAG = 16
"""Bit 4 of a PyMuPDF span's `flags`: the font is bold."""

# Role classification --------------------------------------------------------------------
HEADER_FRAC = 0.12
"""Top/bottom fraction of the page treated as running header/footer (patent headers sit at ~10 %)."""
HEADER_MAX_CHARS = 80
"""Running headers/footers are short; longer text in the margin band is body."""
HEADER_SIZE_TOLERANCE = 0.5
"""Points above body size a header/footer may be; a large title near the top is a heading."""
CAPTION_MAX_CHARS = 400
"""A caption-pattern match longer than this is a paragraph that happens to start with "FIG. n"."""
HEADING_BOLD_SIZE_RATIO = 1.1
"""Bold text at least this much larger than body text is a heading."""
HEADING_BOLD_MAX_CHARS = 60
"""Bold body-size text shorter than this that looks numbered/upper-case is a heading."""
HEADING_SIZE_RATIO = 1.3
"""Text at least this much larger than body text is a heading even when not bold ..."""
HEADING_MAX_CHARS = 120
"""... provided it is shorter than this."""
BODY_SIZE_CHARS_PER_VOTE = 40
"""Body font size is the median over blocks weighted by length: one vote per this many characters."""
HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*\.?\s+\S|[A-Z][A-Z \-&/]{5,}$)")
"""Numbered ("3.2 Thermal") or upper-case ("ELECTRICAL SPECIFICATIONS") heading text."""


class HasBBox(Protocol):
    """Anything placed on a page: raw text blocks and the schema's text/table/figure blocks."""

    @property
    def bbox(self) -> BBox:
        """Position on the page."""
        ...


@dataclass
class RawBlock:
    """A text block before role tagging: geometry, text and the font cues the heuristics use."""

    bbox: BBox
    text: str
    size: float  # dominant font size
    bold: bool
    source: TextSource = "text_layer"
    ocr_confidence: float | None = None  # mean Tesseract word confidence (0-100) for OCR blocks


def raw_blocks_from_page(page: pymupdf.Page) -> list[RawBlock]:
    """Text blocks straight from the PDF text layer (no OCR)."""
    out: list[RawBlock] = []
    for b in page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        if b["type"] != 0:
            continue
        spans = [s for line in b["lines"] for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        lines = [_join_spans(line["spans"]).strip() for line in b["lines"]]
        text = join_lines([line for line in lines if line])
        sizes = [s["size"] for s in spans]
        bold = sum(1 for s in spans if "bold" in s["font"].lower() or s["flags"] & PYMUPDF_BOLD_FLAG) > len(spans) / 2
        x0, y0, x1, y1 = b["bbox"]
        out.append(RawBlock(BBox(x0=x0, y0=y0, x1=x1, y1=y1), text, statistics.median(sizes), bold))
    return out


def _join_spans(spans: list[dict[str, Any]]) -> str:
    """Concatenate a line's spans without inventing spaces.

    MuPDF starts a new span at every font change, so "0.45" set with a different font for
    the period arrives as three spans; joining them with " " produced "0 .45" (seen on an
    IEEE table). A space is inserted only when the spans are physically separated by more
    than SPAN_SPACE_EM and neither side already carries whitespace.
    """
    out = ""
    prev = None
    for sp in spans:
        t = sp["text"]
        if not t:
            continue
        if prev is not None:
            gap = sp["bbox"][0] - prev["bbox"][2]
            if gap > SPAN_SPACE_EM * max(sp["size"], 1.0) and not out.endswith(" ") and not t.startswith(" "):
                out += " "
        out += t
        prev = sp
    return out


def join_lines(lines: list[str]) -> str:
    r"""Join wrapped lines with single spaces, de-hyphenating 'exam-\nple' -> 'example'.

    A trailing hyphen is dropped only when the next line starts lower-case, so part
    numbers broken at a hyphen ("EV-BMS-" / "100") keep it.
    """
    text = ""
    for line in lines:
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        elif text:
            text = text + " " + line
        else:
            text = line
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
        for p in reversed(out[-MERGE_LOOKBACK:]):
            contained = b.bbox.x0 >= p.bbox.x0 - MERGE_X_TOLERANCE and b.bbox.x1 <= p.bbox.x1 + MERGE_X_TOLERANCE
            aligned = abs(p.bbox.x0 - b.bbox.x0) < MERGE_X_TOLERANCE
            gap = b.bbox.y0 - p.bbox.y1
            if (
                (contained or aligned)
                and -MERGE_MAX_OVERLAP <= gap < MERGE_GAP
                and abs(p.size - b.size) < MERGE_SIZE_TOLERANCE
                and not CAPTION_RE.match(b.text)
            ):
                target = p
                break
        if target is not None:
            target.text = join_lines([target.text, b.text])
            target.bbox = BBox(
                x0=min(target.bbox.x0, b.bbox.x0),
                y0=target.bbox.y0,
                x1=max(target.bbox.x1, b.bbox.x1),
                y1=max(target.bbox.y1, b.bbox.y1),
            )
            continue
        out.append(RawBlock(b.bbox, b.text, b.size, b.bold, b.source, b.ocr_confidence))
    return out


def detect_columns(blocks: Sequence[HasBBox], text_x0: float, text_x1: float) -> list[tuple[float, float]]:
    """Return the column x-ranges of a page as (x0, x1) pairs, left to right.

    Columns are the merged x-intervals of the non-spanning blocks; a gap wider than
    MIN_GUTTER separates two columns. With no narrow blocks the whole text area is one column.
    """
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
    return [(a, b) for a, b in merged if (b - a) > MIN_COLUMN_FRAC * width] or [(text_x0, text_x1)]


def order_blocks[T: HasBBox](blocks: Sequence[T], page_height: float) -> list[tuple[T, int | None]]:
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

    def col_of(b: T) -> int | None:
        if b.bbox.width >= FULL_WIDTH_FRAC * width or len(cols) == 1:
            return None if len(cols) > 1 else 0
        cx = b.bbox.cx
        return min(range(len(cols)), key=lambda i: abs((cols[i][0] + cols[i][1]) / 2 - cx))

    tagged = [(b, col_of(b)) for b in blocks]
    spans = sorted([b for b, c in tagged if c is None], key=lambda b: b.bbox.y0)
    # Band boundaries: y0 of each span block.
    bounds = [-1.0] + [s.bbox.y0 for s in spans] + [page_height + 1]
    ordered: list[tuple[T, int | None]] = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if i > 0:
            ordered.append((spans[i - 1], None))
        in_band = [(b, c) for b, c in tagged if c is not None and lo <= b.bbox.y0 < hi]
        for ci in range(len(cols)):
            ordered += sorted([(b, c) for b, c in in_band if c == ci], key=lambda t: t[0].bbox.y0)
    return ordered


def classify_role(block: RawBlock, body_size: float, page_height: float) -> Role:
    """Tag a block as caption, header/footer, heading or body.

    Checked in that order, so a caption in the margin band stays a caption. Header/footer
    needs body-size text: a large title near the top of page 1 is a heading.
    """
    if CAPTION_RE.match(block.text) and len(block.text) < CAPTION_MAX_CHARS:
        return "caption"
    top, bottom = block.bbox.y0 < HEADER_FRAC * page_height, block.bbox.y1 > (1 - HEADER_FRAC) * page_height
    if (top or bottom) and len(block.text) < HEADER_MAX_CHARS and block.size <= body_size + HEADER_SIZE_TOLERANCE:
        return "header" if top else "footer"
    if (block.bold and block.size >= body_size * HEADING_BOLD_SIZE_RATIO) or (
        block.bold and len(block.text) < HEADING_BOLD_MAX_CHARS and HEADING_RE.match(block.text)
    ):
        return "heading"
    if block.size >= body_size * HEADING_SIZE_RATIO and len(block.text) < HEADING_MAX_CHARS:
        return "heading"
    return "body"


def build_text_blocks(raw: list[RawBlock], page_height: float) -> list[TextBlock]:
    """Merge split paragraphs, put them in reading order and tag each with its role and column.

    The body font size the role heuristics compare against is the length-weighted median
    font size of the page, so a page of short headings does not shift it.
    """
    raw = merge_adjacent(raw)
    if not raw:
        return []
    body_size = statistics.median([b.size for b in raw for _ in range(max(1, len(b.text) // BODY_SIZE_CHARS_PER_VOTE))])
    out = []
    for b, col in order_blocks(raw, page_height):
        out.append(
            TextBlock(
                bbox=b.bbox,
                text=b.text,
                role=classify_role(b, body_size, page_height),
                column=col,
                source=b.source,
                ocr_confidence=b.ocr_confidence,
            )
        )
    return out
