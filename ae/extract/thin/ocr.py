"""OCR for scanned pages (thin backend) with Tesseract.

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
from collections import defaultdict

import pymupdf
import pytesseract
from PIL import Image

from ae.extract.thin.text import RawBlock
from ae.schema import BBox

MIN_TEXT_CHARS = 30
IMAGE_COVER_FRAC = 0.5
MIN_TEXT_QUALITY = 0.6
OCR_DPI = 300

_PUNCT = ".,;:!?()[]{}\"'“”‘’«»…-–—/\\|*#&@^~`<>=+"
_NUMERIC = re.compile(r"[\d.,:/%+\-−°×]+[A-Za-zµΩ°]{0,4}")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def text_quality(text: str) -> float:
    """Fraction of tokens that look like real words, numbers or part codes (0..1).

    A token counts as plausible when it is numeric (with an optional short unit),
    an alphanumeric code (120a, M12, J1939), a short all-caps acronym, or an
    alphabetic word containing a vowel. Tokens with U+FFFD or "(cid:" never count.
    Garbled text layers fail because they are vowel-less fragments and symbols.
    """
    tokens = [t.strip(_PUNCT) for t in text.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    good = 0
    for t in tokens:
        if "\ufffd" in t or "(cid:" in t:
            continue
        if _NUMERIC.fullmatch(t):
            good += 1
        elif _WORD.fullmatch(t):
            # Lone letters other than a/I usually mean a text layer split per character.
            if (len(t) == 1 and t in "aAI") or (len(t) > 1 and (re.search(r"[aeiouyAEIOUY]", t) or (t.isupper() and len(t) <= 6))):
                good += 1
        elif re.search(r"[A-Za-z]", t) and re.search(r"\d", t) and len(t) <= 12:
            good += 1
    return good / len(tokens)


def needs_ocr(page: pymupdf.Page) -> str | None:
    """Return the reason this page must be OCR'd, or None if its text layer is usable."""
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


def is_scanned(page: pymupdf.Page) -> bool:
    return needs_ocr(page) is not None


def ocr_page(page: pymupdf.Page, lang: str = "eng") -> list[RawBlock]:
    pix = page.get_pixmap(dpi=OCR_DPI, colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    scale = 72.0 / OCR_DPI
    data = pytesseract.image_to_data(img, lang=lang, config="--psm 3", output_type=pytesseract.Output.DICT)
    # Group words -> lines -> paragraphs (Tesseract 'par' = our block).
    paras: dict[tuple[int, int], dict] = defaultdict(lambda: {"lines": defaultdict(list), "box": None, "conf": []})
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        if not txt or int(data["level"][i]) != 5:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]))
        p = paras[key]
        p["lines"][int(data["line_num"][i])].append((int(data["left"][i]), txt))
        x0, y0 = data["left"][i] * scale, data["top"][i] * scale
        x1, y1 = x0 + data["width"][i] * scale, y0 + data["height"][i] * scale
        b = p["box"]
        p["box"] = (x0, y0, x1, y1) if b is None else (min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1))
        try:
            c = float(data["conf"][i])
            if c >= 0:
                p["conf"].append(c)
        except ValueError:
            pass
    blocks: list[RawBlock] = []
    for key in sorted(paras):
        p = paras[key]
        lines = [" ".join(w for _, w in sorted(ws)) for _, ws in sorted(p["lines"].items())]
        x0, y0, x1, y1 = p["box"]
        line_h = (y1 - y0) / max(1, len(lines))
        from ae.extract.thin.text import _join_lines  # noqa: PLC0415

        blocks.append(
            RawBlock(
                bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                text=_join_lines(lines),
                size=line_h * 0.75,  # approx font size from line height; used only for role heuristics
                bold=False,
                source="ocr",
                conf=(sum(p["conf"]) / len(p["conf"])) if p["conf"] else None,
            )
        )
    return blocks
