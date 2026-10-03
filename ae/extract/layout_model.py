"""Docling's layout model, used for tables and figures on OCR'd PDF pages.

Native extraction cannot find tables or figures on a scanned page: there are no ruling
lines or placed images in a bitmap. Docling's object detector finds them from pixels, so
`ae.extract.base.parse` runs this pass on every PDF that has an OCR'd page and merges the
table and figure blocks of those pages into the native page. Docling's own text is not
used: on the scanned patent it emits header-area regions twice (CER 21.8 % vs 0.13 % for
native's Tesseract text; see docs/extraction.md).

What Docling does per page (v2.x standard PDF pipeline)
-------------------------------------------------------
1. Parse: `docling-parse` (its own C++ parser built on qpdf) extracts text cells
   (words with bboxes and font info) and renders the page bitmap; pypdfium2 is
   the fallback backend.
2. Layout: an object-detection model (default "heron", an RT-DETR style detector
   trained on DocLayNet + internal data) predicts labelled boxes: text,
   section_header, caption, table, picture, list_item, page_header/footer,
   footnote, formula, code. Text cells from step 1 are assigned to boxes by
   overlap; boxes with no cells on a scan are the OCR candidates.
3. OCR: runs on bitmap regions that have no text cells (mode
   PDF_AWARE_LAYOUT_REGIONS). We configure Tesseract CLI, the same engine as native OCR;
   its text fills table cells and the labels inside pictures.
4. Tables: TableFormer (an encoder-decoder transformer) takes the table crop and
   emits an OTSL token sequence (one token per cell: new cell / left-merge /
   up-merge / new line) plus cell bboxes; cell text is then matched back to the
   text cells by bbox overlap.
5. Reading order: a rule-based predictor (`reading_order_rb.py`) builds a
   graph of layout boxes using horizontal/vertical "sees" relations between
   boxes (left-to-right, top-to-bottom, column aware) and topologically sorts
   it; captions are attached to their nearest picture/table.
6. Assemble: a DoclingDocument with provenance (page_no, bbox, char span) on
   every item.

Mapping to our schema: tables -> TableBlock (grid from `TableItem.data`, title = the
nearest section header above it on the page), pictures -> FigureBlock (crop saved from
`get_image`, caption via `caption_text`, id via the shared `ae.extract.captions.figure_id`,
text Docling read inside the picture as labels). Docling bboxes use a bottom-left origin;
we convert to top-left.

Docling is imported lazily (it pulls in torch and the layout models), so documents without
OCR'd pages and every other command never pay for it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pymupdf

from ae.extract.cache import PageCache
from ae.extract.captions import figure_id
from ae.extract.native.figures import FIG_DIR
from ae.log import get_logger
from ae.schema import BBox, FigureBlock, Page, TableBlock

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DoclingDocument

log = get_logger(__name__)

VERSION = "1"
"""Page-cache version of the layout pass; bump it whenever a change alters its blocks."""
PICTURE_IMAGES_SCALE = 2.0
"""Docling renders picture crops at 72 dpi x this scale (~144 dpi)."""
HEADING_LABELS = {"section_header", "title"}
"""Docling item labels whose text becomes the `title` of the tables that follow on the page."""

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    """Return the PDF converter, built once per process (loading the layout and table models is slow)."""
    global _converter
    if _converter is None:
        # Deferred: importing docling loads torch and model code (seconds), see module docstring.
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.ocr_options = TesseractCliOcrOptions(lang=["eng"])
        opts.do_table_structure = True
        # declared as the base options class; the instance is TableStructureOptions, which has the field
        opts.table_structure_options.do_cell_matching = True  # type: ignore[attr-defined]
        opts.generate_picture_images = True
        opts.images_scale = PICTURE_IMAGES_SCALE
        _converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converter


def _bbox(prov: Any, page_h: float) -> BBox:
    """Convert a Docling provenance box (bottom-left origin) to ours (top-left origin)."""
    b = prov.bbox.to_top_left_origin(page_h)
    return BBox(x0=b.l, y0=b.t, x1=b.r, y1=b.b)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _grid_rows(item: Any) -> list[list[str]]:
    """Cell text of a Docling TableItem's grid, all-empty rows dropped."""
    rows = [[_clean(c.text) for c in row] for row in item.data.grid]
    return [r for r in rows if any(r)]


def _save_picture(item: Any, doc: DoclingDocument, name: str) -> Path | None:
    """Write a picture's crop to FIG_DIR/<name>; None when Docling has no image for it."""
    img = item.get_image(doc)
    if img is None:
        return None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    img_path = FIG_DIR / name
    img.save(img_path)
    return img_path


def detect_layout(path: Path, use_cache: bool = True) -> dict[int, Page]:
    """Run Docling on a PDF; return page number -> Page holding only its table and figure blocks.

    Docling converts whole documents, so every page is computed and cached together; the
    document is reconverted only when any page is missing from the cache.
    """
    path = Path(path)
    cache = PageCache("layout", VERSION, path)
    with pymupdf.open(path) as mu:
        n_pages = len(mu)
    if use_cache:
        cached = [cache.get(i + 1) for i in range(n_pages)]
        if all(p is not None for p in cached):
            return {p.number: p for p in cached if p is not None}

    log.info(f"docling layout: converting {path.name}")
    doc = _get_converter().convert(str(path)).document
    pages = {pno: Page(number=pno, width=p.size.width, height=p.size.height) for pno, p in doc.pages.items()}
    _fill_pages(doc, pages, path.stem)
    for p in pages.values():
        cache.put(p)
    return pages


def _fill_pages(doc: DoclingDocument, pages: dict[int, Page], doc_stem: str) -> None:
    """Append every table and picture of `doc` to its page as a TableBlock / FigureBlock."""
    # deferred: see module docstring
    from docling_core.types.doc import ContentLayer, DocItem, PictureItem, TableItem, TextItem

    # Text Docling OCR'd *inside* a picture (reference numerals, box labels) is attached
    # to the picture as child items; it becomes the figure's labels.
    picture_children: dict[str, list[str]] = {}
    for pic in doc.pictures:
        picture_children[pic.self_ref] = [c.resolve(doc).text for c in pic.children if c.cref.startswith("#/texts/")]
    in_picture = {c.cref for pic in doc.pictures for c in pic.children}

    heading: dict[int, str] = {}  # page -> latest section header seen on it (in Docling's reading order)
    fig_i = 0
    for item, _level in doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}):
        if not isinstance(item, DocItem) or not item.prov or item.self_ref in in_picture:
            continue
        prov = item.prov[0]
        page = pages[prov.page_no]
        if isinstance(item, TableItem):
            cap = item.caption_text(doc) or None
            page.blocks.append(
                TableBlock(
                    bbox=_bbox(prov, page.height), rows=_grid_rows(item), caption=cap, title=heading.get(page.number)
                )
            )
        elif isinstance(item, PictureItem):
            fig_i += 1
            img_path = _save_picture(item, doc, f"{doc_stem}_p{prov.page_no}_layout{fig_i}.png")
            cap = item.caption_text(doc) or None
            labels = [t for t in picture_children.get(item.self_ref, []) if t.strip()]
            page.blocks.append(
                FigureBlock(
                    bbox=_bbox(prov, page.height),
                    image_path=str(img_path) if img_path else None,
                    caption=cap,
                    figure_id=figure_id(cap),
                    ocr_labels=labels,
                )
            )
        elif isinstance(item, TextItem) and str(item.label.value) in HEADING_LABELS:
            text = _clean(item.text)
            marker = getattr(item, "marker", None)
            if marker and not text.startswith(marker):
                text = f"{marker} {text}"
            if text:
                heading[page.number] = text
