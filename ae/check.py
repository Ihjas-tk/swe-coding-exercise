"""`ae check`: verify the environment before anything else runs, with one clear line per item."""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from ae import config


def _ok(name: str, detail: str) -> tuple[str, str, str]:
    return ("ok", name, detail)


def _warn(name: str, detail: str) -> tuple[str, str, str]:
    return ("warn", name, detail)


def _fail(name: str, detail: str) -> tuple[str, str, str]:
    return ("fail", name, detail)


def run_checks() -> list[tuple[str, str, str]]:
    out = []
    out.append(_ok("python", sys.version.split()[0]) if sys.version_info[:2] == (3, 12) else _warn("python", f"{sys.version.split()[0]} (pinned: 3.12)"))
    # system packages
    tess = shutil.which("tesseract")
    if tess:
        try:
            v = subprocess.run([tess, "--version"], capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
            langs = subprocess.run([tess, "--list-langs"], capture_output=True, text=True, timeout=10).stdout
            out.append(_ok("tesseract", v) if "eng" in langs else _fail("tesseract", f"{v} but the 'eng' language file is missing"))
        except Exception as e:  # noqa: BLE001
            out.append(_fail("tesseract", f"found but failed to run: {e}"))
    else:
        out.append(_fail("tesseract", "not found (brew install tesseract) — scanned pages and figure labels need it"))
    from ae.extract.pagemap import find_soffice

    so = find_soffice()
    out.append(_ok("libreoffice", so) if so else _warn("libreoffice", "not found (brew install --cask libreoffice) — DOCX pages will be reported as page 1"))
    # python packages
    for mod, label in (("pymupdf", "pymupdf"), ("pdfplumber", "pdfplumber"), ("docling", "docling"), ("sentence_transformers", "sentence-transformers"), ("anthropic", "anthropic")):
        try:
            m = __import__(mod)
            out.append(_ok(label, getattr(m, "__version__", "")))
        except Exception as e:  # noqa: BLE001
            out.append(_fail(label, f"import failed: {e} (run `make setup`)"))
    # API key
    key = config.api_key()
    out.append(_ok("ANTHROPIC_API_KEY", f"set ({len(key)} chars); models: {config.ANSWER_MODEL} / {config.VLM_MODEL}") if key else _warn("ANTHROPIC_API_KEY", "not set — `make ingest` works (no figure descriptions); `make ask` / `make eval` need it (put it in .env)"))
    # models cached?
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    emb = hf / ("models--" + config.EMBED_MODEL.replace("/", "--"))
    out.append(_ok("embedding model", f"{config.EMBED_MODEL} cached") if emb.exists() else _warn("embedding model", f"{config.EMBED_MODEL} not cached — downloaded on first ingest (~300 MB)"))
    dl = [p for p in hf.glob("models--*docling*")] if hf.exists() else []
    out.append(_ok("docling models", f"{len(dl)} cached") if dl else _warn("docling models", "not cached — downloaded on first docling/hybrid ingest (~500 MB)"))
    # indexes
    from ae.index.store import index_db

    for be in ("thin", "docling", "hybrid"):
        db = index_db(be)
        if db.exists():
            try:
                conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                ne = conn.execute("SELECT COUNT(DISTINCT model) FROM embeddings").fetchone()[0]
                conn.close()
                out.append(_ok(f"index {be}", f"{n} chunks, {ne} embedding model(s)") if n else _warn(f"index {be}", "exists but empty — run `make ingest BACKEND=" + be + "`"))
            except sqlite3.OperationalError as e:
                out.append(_warn(f"index {be}", f"unreadable: {e}"))
        else:
            out.append(_warn(f"index {be}", f"missing — `make ingest BACKEND={be}`" + (" (default backend for `make ask`)" if be == config.BACKEND else "")))
    # disk / memory
    try:
        import psutil  # type: ignore

        gb = psutil.virtual_memory().total / 2**30
        out.append(_ok("memory", f"{gb:.0f} GB") if gb >= 12 else _warn("memory", f"{gb:.0f} GB — docling + embeddings on large scans may exceed this"))
    except Exception:  # noqa: BLE001
        pass
    return out


def render(results: list[tuple[str, str, str]]) -> str:
    icon = {"ok": "✓", "warn": "!", "fail": "✗"}
    lines = [f"{icon[s]} {name:20} {detail}" for s, name, detail in results]
    fails = [r for r in results if r[0] == "fail"]
    warns = [r for r in results if r[0] == "warn"]
    lines.append("")
    lines.append(f"{len(fails)} failure(s), {len(warns)} warning(s)" + ("" if not fails else " — fix the failures before `make ingest`"))
    return "\n".join(lines)
