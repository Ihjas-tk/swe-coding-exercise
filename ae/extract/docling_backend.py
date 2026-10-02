"""Docling extraction backend, mapped onto the shared schema.

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
   PDF_AWARE_LAYOUT_REGIONS), using the configured engine. We force Tesseract
   CLI so both backends OCR with the same engine and the comparison isolates
   layout/table/figure handling rather than OCR engines.
4. Tables: TableFormer (an encoder-decoder transformer) takes the table crop and
   emits an OTSL token sequence (one token per cell: new cell / left-merge /
   up-merge / new line) plus cell bboxes; cell text is then matched back to the
   PDF text cells by bbox overlap.
5. Reading order: a rule-based predictor (`reading_order_rb.py`) builds a
   graph of layout boxes using horizontal/vertical "sees" relations between
   boxes (left-to-right, top-to-bottom, column aware) and topologically sorts
   it; captions are attached to their nearest picture/table.
6. Assemble: a DoclingDocument with provenance (page_no, bbox, char span) on
   every item.

Mapping to our schema: texts -> TextBlock (role from Docling's label), tables ->
TableBlock (grid from `TableItem.data`), pictures -> FigureBlock (crop saved from
`get_image`, caption via `caption_text`, id via the shared `ae.extract.captions.figure_id`).
Docling bboxes use a bottom-left origin; we convert to top-left.

Docling is imported lazily (it pulls in torch and the layout models), so the other
backends and the CLI never pay for it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pymupdf

from ae import config
from ae.extract.cache import PageCache
from ae.extract.captions import figure_id
from ae.extract.native.ocr import needs_ocr
from ae.extract.pagemap import BlockPlacer, PageLocator, assemble_pages, render_to_pdf, tables_by_page
from ae.log import get_logger
from ae.schema import BBox, Block, FigureBlock, Page, ParsedDocument, Role, TableBlock, TextBlock

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DoclingDocument

log = get_logger(__name__)

OCR_ENGINE = config.DOCLING_OCR
"""OCR engine Docling drives: "tesseract" (default, same engine as native) or "easyocr" (ablation)."""
VERSION = "4-" + OCR_ENGINE
"""Page-cache version; includes the OCR engine so the ablation never reads the default's pages."""
FIG_DIR = Path("data/extracted/docling/figures")
"""Where this backend writes figure crops."""
PICTURE_IMAGES_SCALE = 2.0
"""Docling renders picture crops at 72 dpi x this scale (~144 dpi)."""
DOCX_PAGE_SIZE = (612.0, 792.0)
"""US Letter in points: page size reported for a DOCX when there is no LibreOffice render."""
DOCX_TEXT_X = (90.0, 522.0)
"""Left/right text edges (1.25 in margins) of synthetic DOCX boxes; Docling exposes no section geometry."""
DOCX_TOP = 72.0
"""Top margin (1 in) where synthetic DOCX boxes start."""

ROLE: dict[str, Role] = {
    "section_header": "heading",
    "title": "heading",
    "caption": "caption",
    "page_header": "header",
    "page_footer": "footer",
    "footnote": "other",
    "formula": "other",
    "code": "other",
}
"""Docling item label -> our block role; every other label (text, list_item, ...) is body."""

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    """Return the PDF converter, built once per process (loading the layout and table models is slow)."""
    global _converter
    if _converter is None:
        # Deferred: importing docling loads torch and model code (seconds), see module docstring.
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, TesseractCliOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = True
        # Tesseract CLI by default so both backends share one OCR engine; EasyOCR (Docling's own
        # default) is the ablation that tests whether region duplication is driver-specific.
        opts.ocr_options = (
            EasyOcrOptions(lang=["en"]) if OCR_ENGINE == "easyocr" else TesseractCliOcrOptions(lang=["eng"])
        )
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


def _caption_refs(doc: DoclingDocument) -> set[str]:
    """Refs of every caption already attached to a table or picture (emitted with it, not as text)."""
    return {ref.cref for ref in _all_caption_refs(doc)}


def _all_caption_refs(doc: DoclingDocument) -> Iterator[Any]:
    for t in doc.tables:
        yield from t.captions
    for p in doc.pictures:
        yield from p.captions


def _save_picture(item: Any, doc: DoclingDocument, name: str) -> Path | None:
    """Write a picture's crop to FIG_DIR/<name>; None when Docling has no image for it."""
    img = item.get_image(doc)
    if img is None:
        return None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    img_path = FIG_DIR / name
    img.save(img_path)
    return img_path


def _nearest_heading(page: Page) -> str | None:
    heads = [b for b in page.blocks if b.kind == "text" and b.role == "heading"]
    return heads[-1].text if heads else None


def _docling_version() -> str:
    try:
        return version("docling")
    except PackageNotFoundError:  # running from a source tree without package metadata
        return "unknown"


# ----------------------------------------------------------------------------- PDF


def parse_pdf(path: Path, use_cache: bool = True) -> ParsedDocument:
    """Parse a PDF with Docling and map it onto the shared schema.

    Pages are cached per page; the document is reconverted only when any page is missing
    (Docling converts whole documents). `is_scanned` uses the native `needs_ocr` test so
    both backends agree on which pages are scans.
    """
    path = Path(path)
    cache = PageCache("docling", VERSION, path)
    with pymupdf.open(path) as mu:
        if use_cache:
            cached = [cache.get(i + 1) for i in range(len(mu))]
            if all(p is not None for p in cached):
                pages = [p for p in cached if p is not None]
                return ParsedDocument(doc=path.name, source_path=str(path), backend="docling", pages=pages)

        log.info(f"docling: converting {path.name}")
        doc = _get_converter().convert(str(path)).document
        by_number = _empty_pages(doc, mu)
    _fill_pdf_pages(doc, by_number, path.stem)

    out_pages = [by_number[k] for k in sorted(by_number)]
    for p in out_pages:
        cache.put(p)
    return ParsedDocument(
        doc=path.name,
        source_path=str(path),
        backend="docling",
        pages=out_pages,
        meta={"docling_version": _docling_version()},
    )


def _empty_pages(doc: DoclingDocument, mu: pymupdf.Document) -> dict[int, Page]:
    """One empty Page per Docling page, flagged scanned by the native test on the same PDF page."""
    pages: dict[int, Page] = {}
    for pno, p in doc.pages.items():
        # "scanned" is a property of the PDF page, so both backends use the same test.
        reason = needs_ocr(mu[pno - 1])
        pages[pno] = Page(
            number=pno,
            width=p.size.width,
            height=p.size.height,
            is_scanned=reason is not None,
            ocr_reason=reason,
            backend="docling",
        )
    return pages


def _fill_pdf_pages(doc: DoclingDocument, pages: dict[int, Page], doc_stem: str) -> None:
    """Append every body/furniture item of `doc` to its page as a text, table or figure block."""
    # deferred: see module docstring
    from docling_core.types.doc import ContentLayer, DocItem, DocItemLabel, PictureItem, TableItem, TextItem

    # Text Docling OCR'd *inside* a picture (reference numerals, box labels) is attached
    # to the picture as child items; collect it as figure labels rather than body text.
    picture_children: dict[str, list[str]] = {}
    for pic in doc.pictures:
        picture_children[pic.self_ref] = [c.resolve(doc).text for c in pic.children if c.cref.startswith("#/texts/")]
    in_picture = {c.cref for pic in doc.pictures for c in pic.children}
    caption_refs = _caption_refs(doc)

    fig_i = 0
    for item, _level in doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}):
        if not isinstance(item, DocItem) or not item.prov or item.self_ref in in_picture:
            continue
        prov = item.prov[0]
        page = pages[prov.page_no]
        bbox = _bbox(prov, page.height)
        if isinstance(item, TableItem):
            cap = item.caption_text(doc) or None
            page.blocks.append(TableBlock(bbox=bbox, rows=_grid_rows(item), caption=cap, title=_nearest_heading(page)))
        elif isinstance(item, PictureItem):
            fig_i += 1
            img_path = _save_picture(item, doc, f"{doc_stem}_p{prov.page_no}_fig{fig_i}.png")
            cap = item.caption_text(doc) or None
            labels = [t for t in picture_children.get(item.self_ref, []) if t.strip()]
            page.blocks.append(
                FigureBlock(
                    bbox=bbox,
                    image_path=str(img_path) if img_path else None,
                    caption=cap,
                    figure_id=figure_id(cap),
                    ocr_labels=labels,
                )
            )
        elif isinstance(item, TextItem):
            if item.label == DocItemLabel.CAPTION and item.self_ref in caption_refs:
                continue  # already attached to its table/figure
            text = _clean(item.text)
            marker = getattr(item, "marker", None)  # list items (patent claims!) keep their number
            if marker and not text.startswith(marker):
                text = f"{marker} {text}"
            if not text:
                continue
            role = ROLE.get(str(item.label.value), "body")
            page.blocks.append(
                TextBlock(bbox=bbox, text=text, role=role, source="ocr" if page.is_scanned else "text_layer")
            )


# ----------------------------------------------------------------------------- DOCX


def parse_docx(path: Path) -> ParsedDocument:
    """Parse a DOCX with Docling's OOXML backend; pages come from ae.extract.pagemap.

    Docling reads the OOXML directly (no layout model, no OCR): paragraphs with styles,
    tables as explicit grids, pictures with their bytes. Like python-docx it has no notion
    of pages (every prov says page 1), so blocks are located in a LibreOffice render.
    """
    from docling.document_converter import DocumentConverter  # deferred: see module docstring
    from docling_core.types.doc import PictureItem, TableItem, TextItem

    path = Path(path)
    log.info(f"docling: converting {path.name}")
    doc = DocumentConverter().convert(str(path)).document
    pdf = render_to_pdf(path)
    locator = PageLocator(pdf) if pdf else None
    placer = BlockPlacer(locator, x0=DOCX_TEXT_X[0], x1=DOCX_TEXT_X[1], top=DOCX_TOP)
    caption_refs = _caption_refs(doc)
    items: list[tuple[int, Block]] = []

    fig_i = tbl_i = 0
    last_heading = None
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            rows = _grid_rows(item)
            tbl_i += 1
            fallback = placer.table_fallback(len(rows))
            items += tables_by_page(rows, locator, last_heading, f"{path.stem}#t{tbl_i}", fallback)
        elif isinstance(item, PictureItem):
            fig_i += 1
            img_path = _save_picture(item, doc, f"{path.stem}_fig{fig_i}.png")
            pno, bbox = placer.place_image()
            cap = item.caption_text(doc) or None
            figure = FigureBlock(
                bbox=bbox, image_path=str(img_path) if img_path else None, caption=cap, figure_id=figure_id(cap)
            )
            items.append((pno, figure))
        elif isinstance(item, TextItem):
            text = _clean(item.text)
            if not text:
                continue
            role = ROLE.get(str(item.label.value), "body")
            if role == "heading":
                last_heading = text
            if role == "caption" and item.self_ref in caption_refs:
                continue
            pno, bbox = placer.place_text(text)
            items.append((pno, TextBlock(bbox=bbox, text=text, role=role)))

    return ParsedDocument(
        doc=path.name,
        source_path=str(path),
        backend="docling",
        pages=assemble_pages(items, locator, DOCX_PAGE_SIZE, "docling"),
        meta={
            "page_mapping": "libreoffice_render" if locator else "unavailable",
            "docling_version": _docling_version(),
        },
    )
