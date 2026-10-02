"""Per-page extraction cache keyed on (backend, backend version, file hash).

Re-running ingest on an unchanged file is free; changing a backend's VERSION
constant invalidates only that backend's entries. This is what keeps a 100-page
PDF tractable during iteration: pages are parsed once.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ae.schema import Page

CACHE_ROOT = Path("data/cache")
"""Root of every on-disk cache (pages, LibreOffice renders, LLM/VLM/embedding stores)."""
HASH_READ_BYTES = 1 << 20
"""Files are hashed in 1 MiB reads, so large PDFs never load into memory at once."""
HASH_HEX_CHARS = 16
"""64 bits of SHA-256: collision-free for any realistic corpus, short enough for directory names."""


def file_hash(path: Path) -> str:
    """Return a short SHA-256 hex digest of the file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(HASH_READ_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()[:HASH_HEX_CHARS]


class PageCache:
    """Per-page JSON cache for one (backend, version, file), under CACHE_ROOT/<backend>/v<version>/."""

    def __init__(self, backend: str, version: str, path: Path) -> None:
        self.dir = CACHE_ROOT / backend / f"v{version}" / f"{path.stem}_{file_hash(path)}"

    def get(self, page_no: int) -> Page | None:
        """Return the cached page, or None."""
        p = self.dir / f"p{page_no}.json"
        if p.exists():
            return Page.model_validate_json(p.read_text())
        return None

    def put(self, page: Page) -> None:
        """Store a parsed page."""
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"p{page.number}.json").write_text(page.model_dump_json())
