"""Per-figure structured descriptions from a multimodal LLM at ingest (optional stage).

Evidence (research notes): text summaries of images retrieve well, but answering visual
questions (counts, locations) from text alone is ~24% worse than from the image, so the
description is indexed for *retrieval* and the crop is still attached at *query* time for
visual questions. The prompt receives the caption and the document's numeral glossary so
the model names parts instead of guessing, lists counted items before giving a number,
and reports uncertainty instead of inventing. Results are cached on image hash + prompt
version, so a 100-page document pays once.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from ae import config
from ae.llm import LLMError, complete_json, image_block
from ae.log import get_logger

log = get_logger(__name__)

PROMPT_VERSION = "v1"
"""Part of the cache key: bump it when SYSTEM changes so old descriptions are not reused."""
CACHE = Path("data/cache/vlm.sqlite")
"""Description cache keyed on image bytes + prompt version + caption + glossary."""
MAX_TOKENS = 1000
"""Reply budget for one figure description."""
ATTEMPTS = 2
"""Tries per figure on timeouts / transient API errors before the figure is skipped."""
MAX_CONNECTIONS_INDEXED = 12
"""Connections kept in the indexed text (long lists add noise, not recall)."""
MAX_VISIBLE_TEXT_INDEXED = 40
"""Visible-text tokens kept in the indexed text."""

SYSTEM = """You describe engineering figures (patent drawings, block diagrams, CAD renderings, photos) for a search index.
Be literal and conservative: report only what is visible. Do not infer function that is not shown.
Reply with ONE JSON object:
{
 "figure_type": "block diagram | schematic | cross-section | photo | rendering | flowchart | other",
 "summary": "one sentence",
 "labelled_parts": [{"label": "160", "name": "control unit", "location": "right, below the main body"}],
 "counts": [{"item": "M12 connector ports", "items_listed": ["top-left", "..."], "count": 4}],
 "connections": ["program drum 62 -> valve control circuit 45"],
 "visible_text": ["any words or numbers printed in the figure"],
 "uncertain": ["labels you could not read or are unsure about"]
}
For labelled_parts use the glossary when a label matches it; if a label is visible but not in the glossary, give name "unknown".
List every counted item in items_listed before stating count. Use [] for empty lists."""


def _cache() -> sqlite3.Connection:
    """Open the description cache, creating it if needed."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE)
    conn.execute("CREATE TABLE IF NOT EXISTS vlm (k TEXT PRIMARY KEY, json TEXT)")
    return conn


def describe_figure(
    image_path: str | Path, caption: str | None, glossary: list[str], model: str | None = None
) -> dict[str, Any] | None:
    """Return the figure's JSON description (cached), or None when the VLM stage cannot run.

    None means no API key is configured or the reply held no JSON object even after the
    strict-JSON retry (LLMError); ingest then stops the VLM stage. A figure whose calls keep failing for transport reasons gets a placeholder
    description with `"_failed": True`, which is not cached, so a re-run retries it.
    """
    if not config.api_key():
        return None
    p = Path(image_path)
    key = hashlib.sha256(
        p.read_bytes() + PROMPT_VERSION.encode() + (caption or "").encode() + "|".join(glossary).encode()
    ).hexdigest()
    with closing(_cache()) as conn:
        row = conn.execute("SELECT json FROM vlm WHERE k = ?", (key,)).fetchone()
        if row:
            log.debug(f"VLM cache hit: {p.name}")
            return json.loads(row[0])
        text = f"Caption: {caption or '(none)'}\nGlossary of reference numerals defined in this document:\n" + (
            "\n".join(glossary) if glossary else "(none)"
        )
        out = None
        for attempt in range(ATTEMPTS):  # retry timeouts / transient API errors, then skip this figure
            try:
                out = complete_json(
                    SYSTEM,
                    [image_block(p), {"type": "text", "text": text}],
                    model=model or config.VLM_MODEL,
                    max_tokens=MAX_TOKENS,
                )
                break
            except LLMError as e:
                if "ANTHROPIC_API_KEY" in str(e):
                    return None  # no key: the whole stage is skipped by the caller
                log.debug(f"VLM call {attempt + 1}/{ATTEMPTS} returned no usable JSON for {p.name}: {e}")
                if attempt == ATTEMPTS - 1:
                    return _failed_description(e)
            # Broad on purpose: SDK transport errors (APITimeoutError, APIConnectionError, rate
            # limits) and malformed JSON (JSONDecodeError) all mean "try again". After the last
            # attempt a placeholder is returned so ingest logs the figure as failed and moves on.
            except Exception as e:
                log.debug(f"VLM call {attempt + 1}/{ATTEMPTS} failed for {p.name}: {type(e).__name__}: {e}")
                if attempt == ATTEMPTS - 1:
                    return _failed_description(e)
        if out is None:
            return None
        conn.execute("INSERT OR REPLACE INTO vlm VALUES (?, ?)", (key, json.dumps(out)))
        conn.commit()
    return out


def _failed_description(error: Exception) -> dict[str, Any]:
    """Return the placeholder description for a figure whose calls kept failing (marked `_failed`)."""
    return {
        "figure_type": "unknown",
        "summary": "",
        "labelled_parts": [],
        "counts": [],
        "connections": [],
        "visible_text": [],
        "uncertain": [f"description failed: {type(error).__name__}"],
        "_failed": True,
    }


def description_text(d: dict[str, Any]) -> str:
    """Flatten a description into the " | "-joined text appended to the figure chunk (what gets indexed)."""
    parts = [d.get("summary", "")]
    if d.get("labelled_parts"):
        parts.append(
            "Parts: " + "; ".join(f"{x.get('label')} = {x.get('name')}" for x in d["labelled_parts"] if x.get("label"))
        )
    if d.get("counts"):
        parts.append("Counts: " + "; ".join(f"{x.get('count')} {x.get('item')}" for x in d["counts"]))
    if d.get("connections"):
        parts.append("Connections: " + "; ".join(d["connections"][:MAX_CONNECTIONS_INDEXED]))
    if d.get("visible_text"):
        parts.append("Visible text: " + " ".join(str(t) for t in d["visible_text"][:MAX_VISIBLE_TEXT_INDEXED]))
    return " | ".join(p for p in parts if p)
