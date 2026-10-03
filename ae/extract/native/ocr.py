"""OCR for scanned pages with Tesseract.

When OCR kicks in
-----------------
Per page, not per document. Two triggers (`needs_ocr`):
1. no_text_layer: fewer than MIN_TEXT_CHARS visible characters and an image
   covering most of the page (a scan).
2. low_text_quality: there *is* a text layer but `text_quality` scores it below
   MIN_TEXT_QUALITY. This catches PDFs whose text layer is garbage: symbol-font
   soup, "(cid:NN)" placeholders, U+FFFD, or a bad prior OCR pass. Such pages
   look born-digital to a character count but would index as noise.
Pages that pass both checks never touch Tesseract.

What Tesseract does (5.x, LSTM engine)
--------------------------------------
1. Binarisation (Otsu) and connected-component analysis find blobs.
2. Page layout analysis (`--psm 3`, fully automatic) groups blobs into text
   lines, detects tab stops / column gutters, and partitions the page into
   blocks; this is what keeps a two-column scan from interleaving.
3. Each line is recognised by the LSTM sequence model with a language model
   (eng.traineddata) producing words with confidences.
`image_to_data` exposes the block/paragraph/line/word hierarchy with pixel
bboxes; we convert blocks to points and hand them to the same reading-order code
as born-digital pages so both paths produce identical structures.
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pymupdf
import pytesseract
from PIL import Image

from ae.extract.native.text import RawBlock, join_lines
from ae.log import get_logger
from ae.schema import BBox

log = get_logger(__name__)

MIN_TEXT_CHARS = 30
"""A page with fewer visible text-layer characters than this may be a scan."""
IMAGE_COVER_FRAC = 0.5
"""... and is one when a single image covers at least this fraction of the page."""
MIN_TEXT_QUALITY = 0.6
"""Text layers whose plausible-token fraction (`text_quality`) is below this are re-OCR'd."""
OCR_DPI = 300
"""Render resolution for page OCR; Tesseract's LSTM model is trained on ~300 dpi text."""
POINTS_PER_INCH = 72.0
"""PDF user-space units per inch (converts rendered pixels back to points)."""
LINE_HEIGHT_TO_FONT_SIZE = 0.75
"""OCR has no font size; ~0.75 x line height approximates it (used only by role heuristics)."""
TESSERACT_PAGE_CONFIG = "--psm 3"
"""Fully automatic page segmentation: finds columns and paragraphs on a scanned page."""
TESSERACT_WORD_LEVEL = 5
"""`image_to_data` row level for a single word (1 page, 2 block, 3 paragraph, 4 line)."""
MAX_CODE_TOKEN_CHARS = 12
"""Mixed letter/digit tokens up to this length count as part codes ("120a", "J1939")."""
MAX_ACRONYM_CHARS = 6
"""Vowel-less all-caps tokens up to this length count as acronyms ("BMS", "CAN")."""

_PUNCT = ".,;:!?()[]{}\"'“”‘’«»…-–—/\\|*#&@^~`<>=+"
_NUMERIC = re.compile(r"[\d.,:/%+\-−°×]+[A-Za-zµΩ°]{0,4}")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

_tesseract_checked = False


def ensure_tesseract() -> None:
    """Raise a clear RuntimeError when the Tesseract binary is missing (checked once per process).

    pytesseract only shells out to `tesseract`; without it every OCR call fails with a
    generic error deep inside a page loop. This turns that into one actionable message.
    """
    global _tesseract_checked
    if _tesseract_checked:
        return
    cmd = pytesseract.pytesseract.tesseract_cmd
    if shutil.which(cmd) is None:
        raise RuntimeError(
            f"Tesseract OCR binary {cmd!r} not found on PATH; it is needed for scanned pages and figure labels. "
            "Install it (macOS: `brew install tesseract`; Debian/Ubuntu: `apt install tesseract-ocr`) "
            "and re-run; `uv run ae check` verifies the toolchain."
        )
    _tesseract_checked = True


def text_quality(text: str) -> float:
    """Fraction of tokens that look like real words, numbers or part codes (0..1).

    A token counts as plausible when it is numeric (with an optional short unit),
    an alphanumeric code (120a, M12, J1939), a short all-caps acronym, or an
    alphabetic word containing a vowel. Tokens with U+FFFD or "(cid:" never count.
    Garbled text layers fail because they are vowel-less fragments and symbols.
    """
    raw_tokens = [t for t in text.split() if t.strip(_PUNCT)]
    if not raw_tokens:
        return 0.0
    # "(cid:NN)" placeholders are checked before punctuation stripping, which would turn them
    # into plausible-looking codes ("cid:56").
    good = sum(1 for t in raw_tokens if "(cid:" not in t and _plausible_token(t.strip(_PUNCT)))
    return good / len(raw_tokens)


def _plausible_token(token: str) -> bool:
    t = token
    if "\ufffd" in t or "(cid:" in t:
        return False
    if _NUMERIC.fullmatch(t):
        return True
    if _WORD.fullmatch(t):
        # Lone letters other than a/I usually mean a text layer split per character.
        return bool(
            (len(t) == 1 and t in "aAI")
            or (len(t) > 1 and (re.search(r"[aeiouyAEIOUY]", t) or (t.isupper() and len(t) <= MAX_ACRONYM_CHARS)))
        )
    return bool(re.search(r"[A-Za-z]", t) and re.search(r"\d", t) and len(t) <= MAX_CODE_TOKEN_CHARS)


def needs_ocr(page: pymupdf.Page) -> str | None:
    """Return why this page must be OCR'd ("no_text_layer" | "low_text_quality"), or None if its text layer is usable.

    A near-empty text layer only means "scan" when a page-covering image exists; a
    genuinely blank page (no characters, no image) is also sent to OCR, which finds nothing.
    """
    text = page.get_text()
    chars = len(text.strip())
    if chars < MIN_TEXT_CHARS:
        page_area = page.rect.width * page.rect.height
        for info in page.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            if (x1 - x0) * (y1 - y0) >= IMAGE_COVER_FRAC * page_area:
                return "no_text_layer"
        return "no_text_layer" if chars == 0 else None
    if text_quality(text) < MIN_TEXT_QUALITY:
        return "low_text_quality"
    return None


@dataclass
class _Paragraph:
    """Words of one Tesseract paragraph, accumulated while scanning `image_to_data` rows."""

    lines: defaultdict[int, list[tuple[int, str]]] = field(default_factory=lambda: defaultdict(list))
    box: tuple[float, float, float, float] | None = None
    confidences: list[float] = field(default_factory=list)


def ocr_page(page: pymupdf.Page, lang: str = "eng") -> list[RawBlock]:
    """OCR a page with Tesseract into paragraph-level RawBlocks (bboxes in points, source "ocr").

    Raises RuntimeError when the Tesseract binary is not installed.
    """
    ensure_tesseract()
    pix = page.get_pixmap(dpi=OCR_DPI, colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    data = pytesseract.image_to_data(img, lang=lang, config=TESSERACT_PAGE_CONFIG, output_type=pytesseract.Output.DICT)
    paragraphs = _group_paragraphs(data, scale=POINTS_PER_INCH / OCR_DPI)
    log.debug(f"OCR page {page.number + 1}: {len(paragraphs)} paragraphs")
    return [_paragraph_block(paragraphs[key]) for key in sorted(paragraphs)]


def _group_paragraphs(data: dict[str, list[Any]], scale: float) -> dict[tuple[int, int], _Paragraph]:
    """Group Tesseract word rows by (block, paragraph); Tesseract's paragraph is our block."""
    paragraphs: dict[tuple[int, int], _Paragraph] = defaultdict(_Paragraph)
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        if not txt or int(data["level"][i]) != TESSERACT_WORD_LEVEL:
            continue
        p = paragraphs[(int(data["block_num"][i]), int(data["par_num"][i]))]
        p.lines[int(data["line_num"][i])].append((int(data["left"][i]), txt))
        x0, y0 = data["left"][i] * scale, data["top"][i] * scale
        x1, y1 = x0 + data["width"][i] * scale, y0 + data["height"][i] * scale
        b = p.box
        p.box = (x0, y0, x1, y1) if b is None else (min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1))
        try:
            c = float(data["conf"][i])
        except ValueError:
            continue  # non-numeric confidence: the word still counts, its confidence does not
        if c >= 0:
            p.confidences.append(c)
    return paragraphs


def _paragraph_block(p: _Paragraph) -> RawBlock:
    lines = [" ".join(w for _, w in sorted(ws)) for _, ws in sorted(p.lines.items())]
    assert p.box is not None  # a paragraph exists only once a word was added to it
    x0, y0, x1, y1 = p.box
    line_h = (y1 - y0) / max(1, len(lines))
    return RawBlock(
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        text=join_lines(lines),
        size=line_h * LINE_HEIGHT_TO_FONT_SIZE,
        bold=False,
        source="ocr",
        ocr_confidence=(sum(p.confidences) / len(p.confidences)) if p.confidences else None,
    )
