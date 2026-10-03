"""Figure extraction and caption linking (native PDF extraction).

How PyMuPDF locates images
--------------------------
`page.get_image_info()` walks the page's display list and reports every raster
image draw command with the bbox it was painted into (in points). This is the
*placed* rectangle, so it is correct even if the image is scaled or rotated.
Vector figures (line art drawn with path operators) have no image object; for
those we use `page.cluster_drawings()`, which groups nearby path bboxes into
rectangles, and keep clusters that look like figures rather than table rules.

Caption linking (ours)
----------------------
Candidate captions are text blocks whose text matches "FIG. N" / "Figure N".
A figure takes the closest candidate that overlaps it horizontally, preferring
one directly below, within MAX_CAPTION_GAP points. Figure ids are normalised by
`ae.extract.captions.figure_id`, shared with the DOCX parser and the layout pass. Captions that sit on the next
page (figure pushed to the bottom of page N, caption at the top of N+1) are
resolved by the orchestrator after all pages are parsed.

Label OCR
---------
Reference numerals ("160", "112") live inside the image pixels. We run Tesseract
in sparse-text mode (psm 11) over the crop and keep confident tokens so that a
figure block is findable by the numerals it contains.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

from ae.extract.captions import FIG_TITLE_RE, figure_id
from ae.extract.native.ocr import ensure_tesseract
from ae.log import get_logger
from ae.schema import BBox, FigureBlock, TableBlock, TextBlock

log = get_logger(__name__)

FIG_DIR = Path("data/extracted/figures")
"""Where figure crops are written (native PDF and DOCX figures, and the layout pass's)."""
MIN_FIGURE_PTS = 60.0
"""Images/drawing clusters smaller than this on a side are icons or rules, not figures."""
MAX_CAPTION_GAP = 90.0
"""Points between a figure edge and its caption block."""
CAPTION_OVERLAP_TOLERANCE = 2.0
"""Points a caption may overlap the figure box and still count as below/above it."""
CAPTION_ABOVE_PENALTY = 30.0
"""Added to the gap of a caption above the figure, so a caption below wins at similar distance."""
FIG_TITLE_MAX_GAP = 30.0
"""A bare "FIG. 2" title this close above a drawing belongs to it."""
CROP_DPI = 200
"""Render resolution of figure crops: enough for label OCR and the VLM, small on disk."""
LABEL_MIN_CONFIDENCE = 50.0
"""Tesseract word confidence (0-100) below which a figure-label token is dropped."""
LABEL_TESSERACT_CONFIG = "--psm 11"
"""Sparse-text segmentation: labels are scattered single words, not paragraphs."""


def find_figure_boxes(page: pymupdf.Page, tables: list[TableBlock]) -> list[BBox]:
    """Raster image boxes + vector-drawing clusters, minus anything that is really a table."""
    boxes: list[BBox] = []
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        b = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
        if b.width >= MIN_FIGURE_PTS and b.height >= MIN_FIGURE_PTS:
            # Not clipped to the text column: on the gripper patent the drawing's right-hand
            # labels ("160") sit a few points past the column edge, and clipping lost them.
            # The price is an occasional sliver of neighbouring text in the label OCR, which
            # the numeral reconciliation ignores.
            boxes.append(b)
    # Vector figures: clusters of drawing paths. Table rulings cluster too, so anything
    # overlapping a detected table is discarded.
    for r in page.cluster_drawings():
        b = BBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)
        if b.width < MIN_FIGURE_PTS or b.height < MIN_FIGURE_PTS:
            continue
        if any(b.overlaps(t.bbox) for t in tables) or any(b.overlaps(o) for o in boxes):
            continue
        boxes.append(b)
    return boxes


def crop_figure(page: pymupdf.Page, bbox: BBox, out_path: Path) -> None:
    """Render the figure region of `page` to a PNG at CROP_DPI (parent dirs are created)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip = pymupdf.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    page.get_pixmap(clip=clip, dpi=CROP_DPI).save(str(out_path))


def ocr_labels(image_path: Path, min_confidence: float = LABEL_MIN_CONFIDENCE) -> list[str]:
    """Return the confident alphanumeric tokens Tesseract reads in a figure image, in reading order.

    Raises RuntimeError when the Tesseract binary is not installed.
    """
    ensure_tesseract()
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, config=LABEL_TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)
    labels = []
    for txt, conf in zip(data["text"], data["conf"], strict=False):
        try:
            c = float(conf)
        except ValueError:
            continue  # unparseable confidence: the token cannot be trusted, skip it
        t = txt.strip()
        if t and c >= min_confidence and re.search(r"[A-Za-z0-9]", t):
            labels.append(t)
    return labels


def link_caption(bbox: BBox, text_blocks: list[TextBlock]) -> TextBlock | None:
    """Return the caption block for a figure box, or None.

    Candidates are caption-role blocks overlapping the figure horizontally within
    MAX_CAPTION_GAP; the nearest wins, with captions above penalised by CAPTION_ABOVE_PENALTY.
    """
    best, best_score = None, float("inf")
    for block in text_blocks:
        if block.role != "caption":
            continue
        h_overlap = min(bbox.x1, block.bbox.x1) - max(bbox.x0, block.bbox.x0)
        if h_overlap <= 0:
            continue
        below = block.bbox.y0 - bbox.y1
        above = bbox.y0 - block.bbox.y1
        is_below = below >= -CAPTION_OVERLAP_TOLERANCE
        gap = below if is_below else above
        if gap < -CAPTION_OVERLAP_TOLERANCE or gap > MAX_CAPTION_GAP:
            continue
        score = gap + (0 if is_below else CAPTION_ABOVE_PENALTY)
        if score < best_score:
            best, best_score = block, score
    return best


def _titles_above(bbox: BBox, text_blocks: list[TextBlock]) -> list[TextBlock]:
    """Bare "FIG. n" title blocks sitting just above the figure and centred over it."""
    return [
        block
        for block in text_blocks
        if FIG_TITLE_RE.match(block.text)
        and 0 <= bbox.y0 - block.bbox.y1 < FIG_TITLE_MAX_GAP
        and block.bbox.cx > bbox.x0
        and block.bbox.cx < bbox.x1
    ]


def extract_figures(
    page: pymupdf.Page,
    text_blocks: list[TextBlock],
    tables: list[TableBlock],
    out_dir: Path,
    doc_stem: str,
    do_ocr_labels: bool = True,
) -> tuple[list[FigureBlock], set[int]]:
    """Crop, caption and label every figure on a born-digital page.

    Returns the figures and the `id()`s of the caption/title text blocks they consumed, so
    the caller can drop those from the page's prose. Crops go to
    `out_dir/<doc_stem>_p<page>_fig<i>.png`.
    """
    figs: list[FigureBlock] = []
    used: set[int] = set()
    for i, bbox in enumerate(find_figure_boxes(page, tables)):
        path = out_dir / f"{doc_stem}_p{page.number + 1}_fig{i + 1}.png"
        crop_figure(page, bbox, path)
        cap = link_caption(bbox, text_blocks)
        if cap is not None:
            used.add(id(cap))
        # A bare "FIG. 2" title directly above the drawing belongs to the figure, not the prose.
        titles = _titles_above(bbox, text_blocks)
        used.update(id(t) for t in titles)
        if cap is None and titles:
            cap = titles[0]
        figs.append(
            FigureBlock(
                bbox=bbox,
                image_path=str(path),
                caption=cap.text if cap else None,
                figure_id=figure_id(cap.text if cap else None),
                ocr_labels=ocr_labels(path) if do_ocr_labels else [],
            )
        )
    log.debug(f"page {page.number + 1}: {len(figs)} figures")
    return figs, used
