"""Page numbers for paginated-on-render formats (DOCX).

A DOCX has no pages: pagination only exists once a layout engine renders it, and
the dev set cites pages of exactly such a render. So we render once with
LibreOffice (headless, open source, local) and then *locate* each extracted
block in the rendered PDF to learn its page and bounding box. Content still
comes from the DOCX XML (exact tables, exact images, exact captions); the render
is consulted only for geometry.

If LibreOffice is not installed, every block is reported on page 1 with a
synthetic bbox and `ParsedDocument.meta["page_mapping"] == "unavailable"`, so the
limitation is visible in the output rather than silent.

`BlockPlacer` and `assemble_pages` are shared by both DOCX parsers (native python-docx
and Docling's OOXML backend), so both locate blocks and fall back identically.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pymupdf

from ae.extract.cache import CACHE_ROOT, file_hash
from ae.log import get_logger
from ae.schema import BBox, Block, Page, TableBlock

log = get_logger(__name__)

SOFFICE_CANDIDATES = ["soffice", "libreoffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice"]
"""LibreOffice binaries tried in order (PATH names, then the default macOS install)."""
RENDER_TIMEOUT_S = 180
"""Seconds a headless LibreOffice conversion may take before it is treated as hung."""

# Locating blocks in the render ------------------------------------------------------------
LOCATE_PREFIX_WORDS = (8, 5, 3)
"""Word-prefix lengths tried, longest first, when searching the render for a block's text."""
MIN_NEEDLE_CHARS = 4
"""Shorter search strings match almost anywhere and are not tried."""
MIN_ROW_NEEDLE_CHARS = 12
"""A table row is located by its longest cell only if that cell is at least this long."""
TABLE_HEADER_ALLOWANCE = 16.0
"""Points added above the first located row of a table piece for its (repeated) header row."""
TABLE_BOTTOM_PAD = 4.0
"""Points added below the last located row of a table piece."""

# Synthetic layout when a block is not found (or there is no render) -----------------------
FLOW_BLOCK_GAP = 6.0
"""Points between consecutive synthetic boxes."""
TEXT_LINE_HEIGHT = 14.0
"""Points per estimated line of a paragraph's synthetic box."""
TEXT_CHARS_PER_LINE = 95
"""Characters per estimated line (a full-width line of 11 pt body text)."""
TABLE_ROW_HEIGHT = 16.0
"""Points per row of a table's synthetic box."""
DEFAULT_IMAGE_HEIGHT = 200.0
"""Points for an image whose size the document does not declare."""


def find_soffice() -> str | None:
    """Path of the LibreOffice binary, or None."""
    for c in SOFFICE_CANDIDATES:
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    return None


def render_to_pdf(path: Path) -> Path | None:
    """Render a DOCX to PDF with LibreOffice and return the PDF path (cached by file hash).

    Returns None, with a warning, when LibreOffice is not installed or produced no PDF;
    callers then fall back to page 1. Raises RuntimeError when the conversion fails or hangs.
    """
    soffice = find_soffice()
    if not soffice:
        log.warning(f"LibreOffice not found: {path.name} will be reported on page 1 (install it for DOCX page numbers)")
        return None
    out_dir = CACHE_ROOT / "render" / f"{path.stem}_{file_hash(path)}"
    pdf = out_dir / f"{path.stem}.pdf"
    if pdf.exists():
        return pdf
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"rendering {path.name} with LibreOffice for page numbers")
    cmd = [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=RENDER_TIMEOUT_S)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode(errors="replace").strip()[-500:]
        raise RuntimeError(
            f"LibreOffice failed to render {path.name} (exit {e.returncode}): {stderr or 'no output'}. "
            f"Check that the file opens in LibreOffice, or run by hand: {' '.join(cmd)}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"LibreOffice did not finish rendering {path.name} within {RENDER_TIMEOUT_S}s. "
            "Close any running LibreOffice window (a running instance blocks headless conversion) and retry."
        ) from e
    if not pdf.exists():
        log.warning(
            f"LibreOffice exited cleanly but wrote no PDF for {path.name}; reporting it on page 1. "
            "A running LibreOffice instance can cause this: close it and re-run."
        )
        return None
    return pdf


class PageLocator:
    """Find where a piece of text (or the i-th image) landed in a rendered PDF.

    Both searches move forward only (document order == page order), so repeated
    phrases resolve to the next occurrence rather than the first one in the file.
    """

    def __init__(self, pdf: Path) -> None:
        self.doc = pymupdf.open(pdf)
        pages = [self.doc[i] for i in range(len(self.doc))]
        self.sizes = [(p.rect.width, p.rect.height) for p in pages]
        self._images = [
            (i + 1, BBox(x0=b[0], y0=b[1], x1=b[2], y1=b[3]))
            for i, p in enumerate(pages)
            for b in (info["bbox"] for info in p.get_image_info())
        ]
        self._img_cursor = 0
        self._cursor_page = 0  # search forward from here: document order == page order

    def locate_text(self, text: str) -> tuple[int, BBox] | None:
        """Return (page, box of the first line) of `text` in the render, or None.

        Tries progressively shorter word prefixes (LOCATE_PREFIX_WORDS), since the render
        re-wraps lines and long needles cross line breaks.
        """
        words = re.sub(r"\s+", " ", text).strip().split(" ")
        for n in LOCATE_PREFIX_WORDS:
            needle = " ".join(words[:n])
            if len(needle) < MIN_NEEDLE_CHARS:
                continue
            for pno in range(self._cursor_page, len(self.doc)):
                hits = self.doc[pno].search_for(needle)
                if hits:
                    r = hits[0]
                    self._cursor_page = pno
                    return pno + 1, BBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)
        return None

    def next_image(self) -> tuple[int, BBox] | None:
        """Page and box of the next image in document order, or None when exhausted."""
        if self._img_cursor < len(self._images):
            hit = self._images[self._img_cursor]
            self._img_cursor += 1
            return hit
        return None


class BlockPlacer:
    """Page number and box for each DOCX block, in document order.

    Blocks found in the render (`locator`) get their rendered page and box. Anything not
    found, and everything when there is no render, gets a synthetic box flowing down
    page 1 from `top` between `x0` and `x1`, so geometry is approximate but never missing.
    """

    def __init__(self, locator: PageLocator | None, x0: float, x1: float, top: float) -> None:
        self.locator = locator
        self.x0, self.x1 = x0, x1
        self._cursor_y = top

    def flow(self, height: float) -> BBox:
        """Next synthetic box of `height` points on page 1."""
        b = BBox(x0=self.x0, y0=self._cursor_y, x1=self.x1, y1=self._cursor_y + height)
        self._cursor_y += height + FLOW_BLOCK_GAP
        return b

    def place_text(self, text: str) -> tuple[int, BBox]:
        """Return the rendered position of a paragraph, else a synthetic box sized from its length."""
        hit = self.locator.locate_text(text) if self.locator else None
        if hit:
            return hit
        return 1, self.flow(TEXT_LINE_HEIGHT * (len(text) // TEXT_CHARS_PER_LINE + 1))

    def place_image(self, height: float = DEFAULT_IMAGE_HEIGHT) -> tuple[int, BBox]:
        """Return the rendered position of the next image, else a synthetic box `height` points tall."""
        hit = self.locator.next_image() if self.locator else None
        return hit if hit else (1, self.flow(height))

    def table_fallback(self, n_rows: int) -> tuple[int, BBox]:
        """Synthetic position for a table; always advances the flow, render or not."""
        return 1, self.flow(TABLE_ROW_HEIGHT * n_rows)


def assemble_pages(
    items: list[tuple[int, Block]], locator: PageLocator | None, fallback_size: tuple[float, float], backend: str
) -> list[Page]:
    """Group (page, block) pairs into Pages sized like the render (one page when there is none)."""
    n_pages = len(locator.sizes) if locator else 1
    pages = []
    for pno in range(1, n_pages + 1):
        w, h = locator.sizes[pno - 1] if locator else fallback_size
        pages.append(Page(number=pno, width=w, height=h, blocks=[b for p, b in items if p == pno], backend=backend))
    return pages


def tables_by_page(
    rows: list[list[str]], locator: PageLocator | None, title: str | None, table_id: str, fallback: tuple[int, BBox]
) -> list[tuple[int, TableBlock]]:
    """Split a table into one TableBlock per rendered page (header row repeated).

    A renderer may break a long table across pages; a citation must then point at
    the page where the *row* actually is, so each row is located individually
    and consecutive rows on the same page form one piece. `fallback` is the
    (page, box) used without a render and for the x-extent of every piece.
    """
    if not rows:
        return []
    if locator is None:
        return [(fallback[0], TableBlock(bbox=fallback[1], rows=rows, title=title, table_id=table_id))]
    header, body = rows[0], rows[1:]
    located = _locate_rows(body, locator, fallback[0])
    pieces: list[tuple[int, TableBlock]] = []
    i = 0
    while i < len(body):
        pno = located[i][0]
        j = i
        while j < len(body) and located[j][0] == pno:
            j += 1
        boxes = [b for _, b in located[i:j] if b is not None]
        x0, x1 = fallback[1].x0, fallback[1].x1
        y0 = min(b.y0 for b in boxes) - TABLE_HEADER_ALLOWANCE if boxes else fallback[1].y0
        y1 = max(b.y1 for b in boxes) + TABLE_BOTTOM_PAD if boxes else fallback[1].y1
        piece = TableBlock(
            bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
            rows=[header, *body[i:j]],
            title=title,
            table_id=table_id,
            continued=i > 0,
        )
        pieces.append((pno, piece))
        i = j
    return pieces


def _locate_rows(body: list[list[str]], locator: PageLocator, first_page: int) -> list[tuple[int, BBox | None]]:
    """(page, box or None) per body row; rows that cannot be found inherit the previous row's page."""
    located: list[tuple[int, BBox | None]] = []
    page = first_page
    for r in body:
        # Locate by the longest cell: first cells are often row numbers, which match anywhere.
        needle = max((c for c in r if c), key=len, default="")
        hit = locator.locate_text(needle) if len(needle) >= MIN_ROW_NEEDLE_CHARS else None
        if hit:
            page = hit[0]
        located.append((page, hit[1] if hit else None))
    return located
