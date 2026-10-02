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
`get_image`, caption via `caption_text`). Docling bboxes use a bottom-left
origin; we convert to top-left.
"""
from __future__ import annotations

import re
from pathlib import Path

from ae.extract.cache import PageCache
from ae.schema import BBox, FigureBlock, Page, ParsedDocument, TableBlock, TextBlock

VERSION = "3"
FIG_DIR = Path("data/extracted/docling/figures")
FIG_ID_RE = re.compile(r"^\s*(FIG(?:URE)?\.?|Figure)\s*(\d+[A-Za-z]?)", re.I)

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.ocr_options = TesseractCliOcrOptions(lang=["eng"])
        opts.do_table_structure = True
        opts.table_structure_options.do_cell_matching = True
        opts.generate_picture_images = True
        opts.images_scale = 2.0  # ~144 dpi crops
        _converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converter


def _bbox(prov, page_h: float) -> BBox:
    b = prov.bbox.to_top_left_origin(page_h)
    return BBox(x0=b.l, y0=b.t, x1=b.r, y1=b.b)


ROLE = {
    "section_header": "heading",
    "title": "heading",
    "caption": "caption",
    "page_header": "header",
    "page_footer": "footer",
    "footnote": "other",
    "formula": "other",
    "code": "other",
}


def parse_pdf(path: Path, use_cache: bool = True) -> ParsedDocument:
    from docling_core.types.doc import DocItemLabel, PictureItem, TableItem, TextItem

    import pymupdf

    path = Path(path)
    cache = PageCache("docling", VERSION, path)

    n_pages = len(pymupdf.open(path))
    if use_cache and all(cache.get(i + 1) is not None for i in range(n_pages)):
        pages = [cache.get(i + 1) for i in range(n_pages)]
        return ParsedDocument(doc=path.name, source_path=str(path), backend="docling", pages=pages)  # type: ignore[arg-type]

    result = _get_converter().convert(str(path))
    doc = result.document
    from ae.extract.thin.ocr import needs_ocr

    mu = pymupdf.open(path)
    pages: dict[int, Page] = {}
    for pno, p in doc.pages.items():
        # "scanned" is a property of the PDF page, so both backends use the same test.
        reason = needs_ocr(mu[pno - 1])
        pages[pno] = Page(
            number=pno, width=p.size.width, height=p.size.height, is_scanned=reason is not None, ocr_reason=reason, backend="docling"
        )

    # Text Docling OCR'd *inside* a picture (reference numerals, box labels) is attached
    # to the picture as child items; collect it as figure labels rather than body text.
    picture_children: dict[str, list[str]] = {}
    for pic in doc.pictures:
        picture_children[pic.self_ref] = [c.resolve(doc).text for c in pic.children if c.cref.startswith("#/texts/")]
    in_picture = {c.cref for pic in doc.pictures for c in pic.children}

    fig_i = 0
    consumed_captions: set[int] = set()
    from docling_core.types.doc import ContentLayer

    for item, _level in doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}):
        if not item.prov or item.self_ref in in_picture:
            continue
        prov = item.prov[0]
        page = pages[prov.page_no]
        bbox = _bbox(prov, page.height)
        if isinstance(item, TableItem):
            grid = item.data.grid
            rows = [[re.sub(r"\s+", " ", c.text).strip() for c in row] for row in grid]
            rows = [r for r in rows if any(r)]
            cap = item.caption_text(doc) or None
            page.blocks.append(TableBlock(bbox=bbox, rows=rows, caption=cap, title=_nearest_heading(page)))
            consumed_captions.update(id(c) for c in item.captions)
        elif isinstance(item, PictureItem):
            fig_i += 1
            img_path = None
            img = item.get_image(doc)
            if img is not None:
                FIG_DIR.mkdir(parents=True, exist_ok=True)
                img_path = FIG_DIR / f"{path.stem}_p{prov.page_no}_fig{fig_i}.png"
                img.save(img_path)
            cap = item.caption_text(doc) or None
            m = FIG_ID_RE.match(cap or "")
            fid = (f"{'FIG.' if m.group(1).upper().startswith('FIG') else 'Figure'} {m.group(2).upper()}") if m else None
            labels = [t for t in picture_children.get(item.self_ref, []) if t.strip()]
            page.blocks.append(
                FigureBlock(bbox=bbox, image_path=str(img_path) if img_path else None, caption=cap, figure_id=fid, ocr_labels=labels)
            )
            consumed_captions.update(id(c) for c in item.captions)
        elif isinstance(item, TextItem):
            if item.label == DocItemLabel.CAPTION and any(ref.cref == item.self_ref for ref in _all_caption_refs(doc)):
                continue  # already attached to its table/figure
            text = re.sub(r"\s+", " ", item.text).strip()
            marker = getattr(item, "marker", None)  # list items (patent claims!) keep their number
            if marker and not text.startswith(marker):
                text = f"{marker} {text}"
            if not text:
                continue
            role = ROLE.get(str(item.label.value), "body")
            page.blocks.append(TextBlock(bbox=bbox, text=text, role=role, source="ocr" if page.is_scanned else "text_layer"))  # type: ignore[arg-type]

    out_pages = [pages[k] for k in sorted(pages)]
    for p in out_pages:
        cache.put(p)
    return ParsedDocument(
        doc=path.name, source_path=str(path), backend="docling", pages=out_pages, meta={"docling_version": _docling_version()}
    )


def _all_caption_refs(doc):
    for t in doc.tables:
        yield from t.captions
    for p in doc.pictures:
        yield from p.captions


def _nearest_heading(page: Page) -> str | None:
    heads = [b for b in page.blocks if b.kind == "text" and b.role == "heading"]
    return heads[-1].text if heads else None


def _docling_version() -> str:
    try:
        from importlib.metadata import version

        return version("docling")
    except Exception:  # pragma: no cover
        return "unknown"


def parse_docx(path: Path, use_cache: bool = True) -> ParsedDocument:
    """Docling's DOCX backend reads the OOXML directly (no layout model, no OCR): paragraphs
    with styles, tables as explicit grids, pictures with their bytes. Like python-docx it has
    no notion of pages (every prov says page 1), so pages come from ae.extract.pagemap."""
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import PictureItem, TableItem, TextItem

    from ae.extract.pagemap import PageLocator, render_to_pdf, tables_by_page

    path = Path(path)
    doc = DocumentConverter().convert(str(path)).document
    pdf = render_to_pdf(path)
    locator = PageLocator(pdf) if pdf else None
    y = 72.0
    items: list[tuple[int, object]] = []

    def flow(h: float) -> BBox:
        nonlocal y
        b = BBox(x0=90, y0=y, x1=522, y1=y + h)
        y += h + 6
        return b

    def place(text: str, h: float) -> tuple[int, BBox]:
        hit = locator.locate_text(text) if locator else None
        return hit if hit else (1, flow(h))

    fig_i = tbl_i = 0
    last_heading = None
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            rows = [[re.sub(r"\s+", " ", c.text).strip() for c in row] for row in item.data.grid]
            rows = [r for r in rows if any(r)]
            tbl_i += 1
            items += tables_by_page(rows, locator, last_heading, f"{path.stem}#t{tbl_i}", (1, flow(16.0 * len(rows))))
        elif isinstance(item, PictureItem):
            fig_i += 1
            img_path = None
            img = item.get_image(doc)
            if img is not None:
                FIG_DIR.mkdir(parents=True, exist_ok=True)
                img_path = FIG_DIR / f"{path.stem}_fig{fig_i}.png"
                img.save(img_path)
            hit = locator.next_image() if locator else None
            pno, bbox = hit if hit else (1, flow(200.0))
            cap = item.caption_text(doc) or None
            m = FIG_ID_RE.match(cap or "")
            fid = (f"{'FIG.' if m.group(1).upper().startswith('FIG') else 'Figure'} {m.group(2).upper()}") if m else None
            items.append((pno, FigureBlock(bbox=bbox, image_path=str(img_path) if img_path else None, caption=cap, figure_id=fid)))
        elif isinstance(item, TextItem):
            text = re.sub(r"\s+", " ", item.text).strip()
            if not text:
                continue
            role = ROLE.get(str(item.label.value), "body")
            if role == "heading":
                last_heading = text
            if role == "caption" and any(ref.cref == item.self_ref for ref in _all_caption_refs(doc)):
                continue
            pno, bbox = place(text, 14.0 * (len(text) // 95 + 1))
            items.append((pno, TextBlock(bbox=bbox, text=text, role=role)))  # type: ignore[arg-type]

    n_pages = len(locator.sizes) if locator else 1
    pages = [
        Page(number=p, width=(locator.sizes[p - 1][0] if locator else 612), height=(locator.sizes[p - 1][1] if locator else 792),
             blocks=[b for pn, b in items if pn == p], backend="docling")  # type: ignore[misc]
        for p in range(1, n_pages + 1)
    ]
    return ParsedDocument(
        doc=path.name, source_path=str(path), backend="docling", pages=pages,
        meta={"page_mapping": "libreoffice_render" if locator else "unavailable", "docling_version": _docling_version()},
    )
