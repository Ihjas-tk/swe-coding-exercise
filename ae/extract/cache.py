"""Per-page extraction cache keyed on (backend, backend version, file hash).

Re-running ingest on an unchanged file is free; changing a backend's VERSION
constant invalidates only that backend's entries. This is what keeps a 100-page
PDF tractable during iteration: pages are parsed once.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ae.schema import Page

CACHE_ROOT = Path("data/cache")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class PageCache:
    def __init__(self, backend: str, version: str, path: Path):
        self.dir = CACHE_ROOT / backend / f"v{version}" / f"{path.stem}_{file_hash(path)}"

    def get(self, page_no: int) -> Page | None:
        p = self.dir / f"p{page_no}.json"
        if p.exists():
            return Page.model_validate_json(p.read_text())
        return None

    def put(self, page: Page) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"p{page.number}.json").write_text(page.model_dump_json())

    def clear(self) -> None:
        if self.dir.exists():
            for f in self.dir.glob("*.json"):
                f.unlink()
