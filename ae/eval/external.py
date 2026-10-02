"""External benchmark preparation: OmniDocBench (English, double-column subset).

OmniDocBench ships page *images*, not PDFs. Each image is wrapped into a
single-page, image-only PDF at the annotated page size so every backend runs on
it through its normal entry point. By construction these pages hit the OCR
path, so this benchmark measures: OCR text accuracy, two-column reading order,
and (for backends that detect tables from pixels) table structure.
"""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf

ROOT = Path("data/external/omnidocbench")
SUBSET = ROOT / "subset_en_double_column.json"
PDF_DIR = ROOT / "pdfs"


def prepare_omnidocbench() -> list[Path]:
    pages = json.loads(SUBSET.read_text())
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for pg in pages:
        info = pg["page_info"]
        img = ROOT / "images" / info["image_path"]
        pdf = PDF_DIR / (Path(info["image_path"]).stem + ".pdf")
        if not pdf.exists():
            # Letter-ish page at 72 pt/in; keep the image aspect ratio, 8.5in wide.
            w_pt = 612.0
            h_pt = w_pt * info["height"] / info["width"]
            doc = pymupdf.open()
            page = doc.new_page(width=w_pt, height=h_pt)
            page.insert_image(page.rect, filename=str(img))
            doc.save(str(pdf))
        out.append(pdf)
    return out
