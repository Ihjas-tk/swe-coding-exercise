"""Which files make up the corpus: the built-in directories or a directory given explicitly."""

from __future__ import annotations

from pathlib import Path

from ae import config

CORPUS_DIRS = ("patents", "design_docs", "structured")
"""The built-in corpus, relative to the project root."""
DOCUMENT_SUFFIXES = (".pdf", ".docx")
"""Files that go through extraction (ae.extract.base.parse)."""
STRUCTURED_SUFFIXES = (".csv", ".xlsx")
"""Files that go through the structured loader (ae.extract.structured)."""
SUPPORTED = DOCUMENT_SUFFIXES + STRUCTURED_SUFFIXES


def corpus_files(root: Path = Path("."), corpus: Path | None = None) -> list[Path]:
    """Files to ingest, sorted: every supported file under `corpus` (recursive), else the built-in corpus dirs.

    `corpus` defaults to AE_CORPUS. The built-in dirs are listed non-recursively and
    unfiltered (dotfiles aside), so an unsupported file there surfaces as an error.
    """
    corpus = corpus or (Path(config.CORPUS) if config.CORPUS else None)
    if corpus:
        return sorted(
            p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith(".")
        )
    files: list[Path] = []
    for d in CORPUS_DIRS:
        files += sorted(p for p in (root / d).glob("*") if p.is_file() and not p.name.startswith("."))
    return files
