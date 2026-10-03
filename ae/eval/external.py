"""External evaluation data: the question-set corpus download and OmniDocBench preparation.

`fetch_corpus` downloads the files listed in gold/external_manifest.json (sha256-checked;
a changed checksum is reported, not fatal: publishers update files) plus selected ICDAR
2013 table-competition PDFs from their zip.

OmniDocBench ships page *images*, not PDFs. Each image is wrapped into a
single-page, image-only PDF at the annotated page size so the pipeline runs on
it through its normal entry point. By construction these pages hit the OCR
path, so this benchmark measures: OCR text accuracy, two-column reading order,
table structure and figure detection from pixels (the Docling layout pass).
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pymupdf

from ae.log import get_logger

log = get_logger(__name__)

ROOT = Path("data/external/omnidocbench")
SUBSET = ROOT / "subset_en_double_column.json"
PDF_DIR = ROOT / "pdfs"
ICDAR_RAW = Path("data/external/icdar2013_raw")
USER_AGENT = {"User-Agent": "Mozilla/5.0"}  # some publishers reject urllib's default agent
FILE_TIMEOUT_S = 120
ZIP_TIMEOUT_S = 300


def fetch_corpus(manifest: Path) -> Path:
    """Download every manifest file (skipping ones already present with the right sha256) and the ICDAR picks."""
    man = json.loads(manifest.read_text())
    root = Path(man["root"])
    for f in man["files"]:
        dst = root / f["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == f["sha256"]:
            log.info(f"ok {f['path']}")
            continue
        data = _download(f["url"], FILE_TIMEOUT_S)
        dst.write_bytes(data)
        if hashlib.sha256(data).hexdigest() == f["sha256"]:
            log.info(f"fetched {f['path']}")
        else:
            log.warning(f"fetched {f['path']} (checksum differs: publisher updated the file)")
    _fetch_icdar(man["icdar2013"], root / "icdar2013")
    return root


def _fetch_icdar(ic: dict, dest: Path) -> None:
    """Unpack the ICDAR 2013 competition zip once and copy the selected PDFs to `dest`."""
    if not ICDAR_RAW.exists():
        zipfile.ZipFile(io.BytesIO(_download(ic["zip"], ZIP_TIMEOUT_S))).extractall(ICDAR_RAW)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ic["selected"]:
        d = name.replace("icdar2013_", "").replace(".pdf", "")
        src = ICDAR_RAW / f"competition-dataset-{d.split('-')[0]}" / f"{d}.pdf"
        if src.exists() and not (dest / name).exists():
            shutil.copy(src, dest / name)


def _download(url: str, timeout: int) -> bytes:
    """GET a URL with a browser user agent."""
    req = urllib.request.Request(url, headers=USER_AGENT)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def prepare_omnidocbench() -> list[Path]:
    """Wrap each subset page image into a single-page PDF; returns the PDF paths."""
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
