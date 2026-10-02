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
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pymupdf

from ae.extract.cache import CACHE_ROOT, file_hash
from ae.schema import BBox, TableBlock

SOFFICE_CANDIDATES = ["soffice", "libreoffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice"]


def find_soffice() -> str | None:
    for c in SOFFICE_CANDIDATES:
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    return None


def render_to_pdf(path: Path) -> Path | None:
    """Render a DOCX to PDF with LibreOffice; cached by file hash. None if unavailable."""
    soffice = find_soffice()
    if not soffice:
        return None
    out_dir = CACHE_ROOT / "render" / f"{path.stem}_{file_hash(path)}"
    pdf = out_dir / f"{path.stem}.pdf"
    if pdf.exists():
        return pdf
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(path)],
        check=True,
        capture_output=True,
        timeout=180,
    )
    return pdf if pdf.exists() else None


class PageLocator:
    """Find where a piece of text (or the i-th image) landed in a rendered PDF."""

    def __init__(self, pdf: Path):
        self.doc = pymupdf.open(pdf)
        self.sizes = [(p.rect.width, p.rect.height) for p in self.doc]
        self._images = [(i + 1, BBox(x0=b[0], y0=b[1], x1=b[2], y1=b[3])) for i, p in enumerate(self.doc) for b in (info["bbox"] for info in p.get_image_info())]
        self._img_cursor = 0
        self._cursor_page = 0  # search forward from here: document order == page order

    def locate_text(self, text: str) -> tuple[int, BBox] | None:
        """Try progressively shorter prefixes of the text until the render contains it."""
        words = re.sub(r"\s+", " ", text).strip().split(" ")
        for n in (8, 5, 3):
            needle = " ".join(words[:n])
            if len(needle) < 4:
                continue
            for pno in range(self._cursor_page, len(self.doc)):
                hits = self.doc[pno].search_for(needle)
                if hits:
                    r = hits[0]
                    self._cursor_page = pno
                    return pno + 1, BBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)
        return None

    def next_image(self) -> tuple[int, BBox] | None:
        if self._img_cursor < len(self._images):
            hit = self._images[self._img_cursor]
            self._img_cursor += 1
            return hit
        return None


def tables_by_page(rows: list[list[str]], locator: PageLocator | None, title: str | None, table_id: str, fallback: tuple[int, BBox]) -> list[tuple[int, TableBlock]]:
    """Split a table into one TableBlock per rendered page (header row repeated).

    A renderer may break a long table across pages; a citation must then point at
    the page where the *row* actually is, so each row is located individually
    and consecutive rows on the same page form one piece.
    """
    if not rows:
        return []
    if locator is None:
        return [(fallback[0], TableBlock(bbox=fallback[1], rows=rows, title=title, table_id=table_id))]
    header, body = rows[0], rows[1:]
    located: list[tuple[int, BBox | None]] = []
    page = fallback[0]
    for r in body:
        hit = locator.locate_text(next((c for c in r if c), "")) if any(r) else None
        if hit:
            page = hit[0]
        located.append((page, hit[1] if hit else None))
    pieces: list[tuple[int, TableBlock]] = []
    i = 0
    while i < len(body):
        pno = located[i][0]
        j = i
        while j < len(body) and located[j][0] == pno:
            j += 1
        boxes = [b for _, b in located[i:j] if b is not None]
        x0, x1 = fallback[1].x0, fallback[1].x1
        y0 = min(b.y0 for b in boxes) - 16 if boxes else fallback[1].y0
        y1 = max(b.y1 for b in boxes) + 4 if boxes else fallback[1].y1
        pieces.append((pno, TableBlock(bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), rows=[header] + body[i:j], title=title, table_id=table_id, continued=i > 0)))
        i = j
    return pieces
