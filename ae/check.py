"""`ae check`: verify the environment before anything else runs, with one clear line per item."""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]


def _find_dotenv() -> Path | None:
    """Locate the .env the same way `ae.config.load_dotenv` does (CWD, then each parent)."""
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / ".env").exists():
            return d / ".env"
    return None


def _check_api_live() -> tuple[str, str, str]:
    """One five-token live call: catches an invalid key or an exhausted credit balance before a long run."""
    from ae.llm import complete

    try:
        complete("Reply with the single word OK.", "ping", max_tokens=5, use_cache=False)
        return _ok("API", "reachable (one live test call succeeded)")
    except Exception as e:  # any SDK/transport error: report it verbatim, the user must act on it
        msg = str(e)
        if "credit balance" in msg:
            return _fail(
                "API", "the key's credit balance is exhausted — top up at console.anthropic.com or use another key"
            )
        return _fail("API", f"live call failed: {type(e).__name__}: {msg[:160]}")


def _check_dotenv() -> tuple[str, str, str]:
    env = _find_dotenv()
    if env:
        return _ok(".env", str(env))
    if config.api_key():
        return _ok(".env", "none found; ANTHROPIC_API_KEY comes from the environment")
    return _warn(".env", f"missing — run `cp .env.example .env` in {ROOT} and set ANTHROPIC_API_KEY in it")


def _check_uv() -> list[tuple[str, str, str]]:
    """uv.lock agrees with pyproject.toml, and the venv matches uv.lock (extra packages allowed)."""
    uv = shutil.which("uv")
    if not uv:
        return [_warn("uv.lock", "uv not on PATH — cannot verify the environment against uv.lock")]
    if not (ROOT / "uv.lock").exists():
        return [_fail("uv.lock", f"missing in {ROOT} — run `uv lock && uv sync`")]

    def uv_run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([uv, *args, "--offline"], cwd=ROOT, capture_output=True, text=True, timeout=60)

    try:
        lock = uv_run("lock", "--check")
        if lock.returncode != 0:
            return [_warn("uv.lock", "out of date with pyproject.toml — run `uv lock && uv sync`")]
        sync = uv_run("sync", "--check", "--inexact")
    except Exception as e:  # timeouts, old uv without --check
        return [_warn("uv.lock", f"could not verify: {e}")]
    if sync.returncode != 0:
        todo = [ln.strip() for ln in (sync.stdout + sync.stderr).splitlines() if ln.strip().startswith(("+", "-", "~"))]
        return [_warn("uv.lock", f"environment differs from uv.lock ({len(todo) or '?'} package(s)) — run `uv sync`")]
    return [_ok("uv.lock", "up to date; environment in sync")]


def run_checks() -> list[tuple[str, str, str]]:
    """Return (status, name, detail) for each environment check."""
    out = []
    out.append(
        _ok("python", sys.version.split()[0])
        if sys.version_info[:2] == (3, 12)
        else _warn("python", f"{sys.version.split()[0]} (pinned: 3.12)")
    )
    # system packages
    tess = shutil.which("tesseract")
    if tess:
        try:
            v = subprocess.run([tess, "--version"], capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
            langs = subprocess.run([tess, "--list-langs"], capture_output=True, text=True, timeout=10).stdout
            out.append(
                _ok("tesseract", v)
                if "eng" in langs
                else _fail("tesseract", f"{v} but the 'eng' language file is missing")
            )
        except Exception as e:  # any failure to run tesseract is reported, not raised
            out.append(_fail("tesseract", f"found but failed to run: {e}"))
    else:
        out.append(_fail("tesseract", "not found (brew install tesseract) — scanned pages and figure labels need it"))
    from ae.extract.pagemap import find_soffice

    so = find_soffice()
    out.append(
        _ok("libreoffice", so)
        if so
        else _warn("libreoffice", "not found (brew install --cask libreoffice) — DOCX pages will be reported as page 1")
    )
    # python packages
    for mod, label in (
        ("pymupdf", "pymupdf"),
        ("pdfplumber", "pdfplumber"),
        ("docling", "docling"),
        ("sentence_transformers", "sentence-transformers"),
        ("anthropic", "anthropic"),
    ):
        try:
            m = __import__(mod)
            out.append(_ok(label, getattr(m, "__version__", "")))
        except Exception as e:  # import errors of native extensions are not always ImportError
            out.append(_fail(label, f"import failed: {e} (run `make setup`)"))
    out.extend(_check_uv())
    # API key
    out.append(_check_dotenv())
    key = config.api_key()
    out.append(
        _ok("ANTHROPIC_API_KEY", f"set ({len(key)} chars); models: {config.ANSWER_MODEL} / {config.VLM_MODEL}")
        if key
        else _warn(
            "ANTHROPIC_API_KEY",
            "not set — `make ingest` works (no figure descriptions); `make ask` / `make eval` need it (put it in .env)",
        )
    )
    if key:
        out.append(_check_api_live())
    # models cached? (same cache directory huggingface_hub resolves: HF_HUB_CACHE, else HF_HOME/hub)
    from ae.models import DOCLING_REPOS, hub_cache, is_cached

    out.append(
        _ok("embedding model", f"{config.EMBED_MODEL} cached")
        if is_cached(config.EMBED_MODEL)
        else _warn(
            "embedding model",
            f"{config.EMBED_MODEL} not cached in {hub_cache()} — `make models` (or first ingest) downloads ~300 MB",
        )
    )
    dl = [r for r in DOCLING_REPOS if is_cached(r)]
    out.append(
        _ok("docling models", f"{len(dl)} cached")
        if len(dl) == len(DOCLING_REPOS)
        else _warn(
            "docling models",
            f"{len(dl)}/{len(DOCLING_REPOS)} cached — `make models` (or the first ingest of a scanned PDF) downloads ~500 MB",
        )
    )
    # index
    from ae.answer.pipeline import ingest_hint
    from ae.index.store import index_db

    db = index_db()
    if db.exists():
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            ne = conn.execute("SELECT COUNT(DISTINCT model) FROM embeddings").fetchone()[0]
            conn.close()
            out.append(
                _ok("index", f"{db}: {n} chunks, {ne} embedding model(s)")
                if n
                else _warn("index", f"{db} exists but is empty — run {ingest_hint()}")
            )
        except sqlite3.OperationalError as e:
            out.append(_warn("index", f"{db} unreadable: {e}"))
    else:
        out.append(_warn("index", f"{db} missing — run {ingest_hint()} (needed by `make ask` / `make eval`)"))
    # disk / memory
    try:
        import psutil

        gb = psutil.virtual_memory().total / 2**30
        out.append(
            _ok("memory", f"{gb:.0f} GB")
            if gb >= 12
            else _warn("memory", f"{gb:.0f} GB — docling + embeddings on large scans may exceed this")
        )
    except Exception:  # psutil is optional
        pass
    return out


def render(results: list[tuple[str, str, str]]) -> str:
    """Format check results as one line each plus a totals line."""
    icon = {"ok": "✓", "warn": "!", "fail": "✗"}
    lines = [f"{icon[s]} {name:20} {detail}" for s, name, detail in results]
    fails = [r for r in results if r[0] == "fail"]
    warns = [r for r in results if r[0] == "warn"]
    lines.append("")
    lines.append(
        f"{len(fails)} failure(s), {len(warns)} warning(s)"
        + ("" if not fails else " — fix the failures before `make ingest`")
    )
    return "\n".join(lines)
