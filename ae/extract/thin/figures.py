"""Figure extraction and caption linking (thin backend).

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
one directly below, within MAX_CAPTION_GAP points. Captions that sit on the next
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

from ae.extract.thin.text import FIG_TITLE_RE
from ae.schema import BBox, FigureBlock, TableBlock, TextBlock

MIN_FIGURE_PTS = 60.0  # ignore images/clusters smaller than this on a side (icons, rules)
MAX_CAPTION_GAP = 90.0  # points between figure edge and caption block
FIG_ID_RE = re.compile(r"^\s*(FIG(?:URE)?\.?|Figure)\s*(\d+[A-Za-z]?)", re.I)
CROP_DPI = 200


def find_figure_boxes(page: pymupdf.Page, tables: list[TableBlock], columns: list[tuple[float, float]]) -> list[BBox]:
    """Raster image boxes + vector-drawing clusters, minus anything that is really a table.

    `columns` are the page's text column x-ranges (from text.detect_columns); an
    image that is placed wider than its column (transparent margins) is clipped to
    the column so the crop does not pick up neighbouring text.
    """
    boxes: list[BBox] = []
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        b = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
        if b.width >= MIN_FIGURE_PTS and b.height >= MIN_FIGURE_PTS:
            boxes.append(_clip_to_column(b, columns))
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


def _clip_to_column(b: BBox, columns: list[tuple[float, float]]) -> BBox:
    if len(columns) < 2:
        return b
    # Column the image centre falls in; only clip if the image mostly lives in one column.
    col = min(columns, key=lambda c: abs((c[0] + c[1]) / 2 - b.cx))
    if b.width > 0.8 * (columns[-1][1] - columns[0][0]):
        return b
    return BBox(x0=max(b.x0, col[0] - 6), y0=b.y0, x1=min(b.x1, col[1] + 6), y1=b.y1)


def crop_figure(page: pymupdf.Page, bbox: BBox, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip = pymupdf.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    page.get_pixmap(clip=clip, dpi=CROP_DPI).save(str(out_path))


def ocr_labels(image_path: Path, min_conf: float = 50.0) -> list[str]:
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, config="--psm 11", output_type=pytesseract.Output.DICT)
    labels = []
    for txt, conf in zip(data["text"], data["conf"]):
        try:
            c = float(conf)
        except ValueError:
            continue
        t = txt.strip()
        if t and c >= min_conf and re.search(r"[A-Za-z0-9]", t):
            labels.append(t)
    return labels


def link_caption(bbox: BBox, text_blocks: list[TextBlock]) -> TextBlock | None:
    best, best_score = None, float("inf")
    for tb in text_blocks:
        if tb.role != "caption":
            continue
        h_overlap = min(bbox.x1, tb.bbox.x1) - max(bbox.x0, tb.bbox.x0)
        if h_overlap <= 0:
            continue
        below = tb.bbox.y0 - bbox.y1
        above = bbox.y0 - tb.bbox.y1
        gap = below if below >= -2 else above
        if gap < -2 or gap > MAX_CAPTION_GAP:
            continue
        score = gap + (0 if below >= -2 else 30)  # prefer captions below
        if score < best_score:
            best, best_score = tb, score
    return best


def figure_id(caption: str | None) -> str | None:
    if not caption:
        return None
    m = FIG_ID_RE.match(caption)
    if not m:
        return None
    word = "FIG." if m.group(1).upper().startswith("FIG.") or m.group(1).upper() == "FIG" else "Figure"
    return f"{word} {m.group(2).upper()}"


def extract_figures(
    page: pymupdf.Page,
    text_blocks: list[TextBlock],
    tables: list[TableBlock],
    columns: list[tuple[float, float]],
    out_dir: Path,
    doc_stem: str,
    do_ocr_labels: bool = True,
) -> tuple[list[FigureBlock], set[int]]:
    """Returns figures and the ids() of caption/title text blocks consumed by them."""
    figs: list[FigureBlock] = []
    used: set[int] = set()
    for i, bbox in enumerate(find_figure_boxes(page, tables, columns)):
        path = out_dir / f"{doc_stem}_p{page.number + 1}_fig{i + 1}.png"
        crop_figure(page, bbox, path)
        cap = link_caption(bbox, text_blocks)
        if cap is not None:
            used.add(id(cap))
        # A bare "FIG. 2" title directly above the drawing belongs to the figure, not the prose.
        for tb in text_blocks:
            if FIG_TITLE_RE.match(tb.text) and 0 <= bbox.y0 - tb.bbox.y1 < 30 and tb.bbox.cx > bbox.x0 and tb.bbox.cx < bbox.x1:
                used.add(id(tb))
                if cap is None:
                    cap = tb
        figs.append(
            FigureBlock(
                bbox=bbox,
                image_path=str(path),
                caption=cap.text if cap else None,
                figure_id=figure_id(cap.text if cap else None),
                ocr_labels=ocr_labels(path) if do_ocr_labels else [],
            )
        )
    return figs, used
