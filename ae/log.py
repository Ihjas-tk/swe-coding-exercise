"""Logging and timing for every stage.

- `get_logger(name)` returns a logger that writes INFO+ to stderr (level from AE_LOG, default
  INFO) and DEBUG+ to data/logs/ae.log (rotating, 5 MB x 3).
- `timed(stage, doc=None, **fields)` is a context manager that logs the duration and records
  it in the process-wide TIMINGS list; `flush_timings(path)` writes them as JSON and
  `timing_table()` renders a per-document summary. Query latency is recorded the same way
  under doc=None with stage names prefixed "query.".
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("data/logs")
_configured = False
TIMINGS: list[dict] = []


def _configure() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    root = logging.getLogger("ae")
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, os.environ.get("AE_LOG", "INFO").upper(), logging.INFO))
    console.setFormatter(fmt)
    root.addHandler(console)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(LOG_DIR / "ae.log", maxBytes=5_000_000, backupCount=3)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(fh)
    except OSError:
        pass
    # quieten noisy third parties
    for noisy in ("httpx", "urllib3", "sentence_transformers", "transformers", "docling", "torch", "PIL", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name if name.startswith("ae") else f"ae.{name}")


@contextmanager
def timed(stage: str, doc: str | None = None, level: int = logging.INFO, **fields):
    log = get_logger("ae.timing")
    t0 = time.perf_counter()
    try:
        yield fields
    finally:
        ms = int((time.perf_counter() - t0) * 1000)
        rec = {"stage": stage, "doc": doc, "ms": ms, **fields}
        TIMINGS.append(rec)
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        log.log(level, f"{stage}{' ' + doc if doc else ''}: {ms} ms{(' ' + extra) if extra else ''}")


def flush_timings(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(TIMINGS, indent=1))


def timing_table(stages: tuple[str, ...] = ("extract", "vlm", "chunk", "embed")) -> str:
    """Per-document table of the main ingest stages (ms), plus totals."""
    docs: dict[str, dict[str, int]] = {}
    for r in TIMINGS:
        if r["doc"] and r["stage"] in stages:
            docs.setdefault(r["doc"], {})[r["stage"]] = docs.get(r["doc"], {}).get(r["stage"], 0) + r["ms"]
    if not docs:
        return ""
    lines = ["| document | " + " | ".join(f"{s} ms" for s in stages) + " |", "|---|" + "---|" * len(stages)]
    for d, st in docs.items():
        lines.append(f"| {d[:48]} | " + " | ".join(str(st.get(s, "")) for s in stages) + " |")
    tot = {s: sum(st.get(s, 0) for st in docs.values()) for s in stages}
    lines.append("| **total** | " + " | ".join(str(tot[s]) for s in stages) + " |")
    other = {r["stage"]: r["ms"] for r in TIMINGS if not r["doc"] and not r["stage"].startswith("query.")}
    if other:
        lines.append("")
        lines.append("Corpus-level stages: " + ", ".join(f"{k} {v} ms" for k, v in other.items()))
    return "\n".join(lines)


def latency_summary(prefix: str = "query.") -> dict:
    """p50/p95 per query stage over recorded timings."""
    by: dict[str, list[int]] = {}
    for r in TIMINGS:
        if r["stage"].startswith(prefix):
            by.setdefault(r["stage"][len(prefix):], []).append(r["ms"])
    out = {}
    for k, v in by.items():
        v = sorted(v)
        out[k] = {"n": len(v), "p50": v[len(v) // 2], "p95": v[min(len(v) - 1, int(len(v) * 0.95))], "max": v[-1]}
    return out
