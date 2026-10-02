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
from pathlib import Path

from ae import config
from ae.llm import LLMError, complete_json, image_block
from ae.log import get_logger, timed

log = get_logger(__name__)

PROMPT_VERSION = "v1"
CACHE = Path("data/cache/vlm.sqlite")

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


def _cache():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE)
    conn.execute("CREATE TABLE IF NOT EXISTS vlm (k TEXT PRIMARY KEY, json TEXT)")
    return conn


def describe_figure(image_path: str | Path, caption: str | None, glossary: list[str], model: str | None = None) -> dict | None:
    """Returns the parsed JSON description, or None when no API key is configured."""
    if not config.api_key():
        return None
    p = Path(image_path)
    key = hashlib.sha256(p.read_bytes() + PROMPT_VERSION.encode() + (caption or "").encode() + "|".join(glossary).encode()).hexdigest()
    conn = _cache()
    row = conn.execute("SELECT json FROM vlm WHERE k = ?", (key,)).fetchone()
    if row:
        conn.close()
        return json.loads(row[0])
    text = f"Caption: {caption or '(none)'}\nGlossary of reference numerals defined in this document:\n" + ("\n".join(glossary) if glossary else "(none)")
    out = None
    for attempt in range(2):  # one retry on timeouts / transient API errors, then skip this figure
        try:
            with timed("vlm.call", level=10, image=p.name):
                out = complete_json(SYSTEM, [image_block(p), {"type": "text", "text": text}], model=model or config.VLM_MODEL, max_tokens=1000)
            break
        except LLMError:
            conn.close()
            return None
        except Exception as e:  # noqa: BLE001  (APITimeoutError, connection errors, bad JSON)
            if attempt == 1:
                conn.close()
                return {"figure_type": "unknown", "summary": "", "labelled_parts": [], "counts": [], "connections": [], "visible_text": [], "uncertain": [f"description failed: {type(e).__name__}"], "_failed": True}
    if out is None:
        conn.close()
        return None
    conn.execute("INSERT OR REPLACE INTO vlm VALUES (?, ?)", (key, json.dumps(out)))
    conn.commit()
    conn.close()
    return out


def description_text(d: dict) -> str:
    """Flatten the JSON for the figure chunk (what gets indexed)."""
    parts = [d.get("summary", "")]
    if d.get("labelled_parts"):
        parts.append("Parts: " + "; ".join(f"{x.get('label')} = {x.get('name')}" for x in d["labelled_parts"] if x.get("label")))
    if d.get("counts"):
        parts.append("Counts: " + "; ".join(f"{x.get('count')} {x.get('item')}" for x in d["counts"]))
    if d.get("connections"):
        parts.append("Connections: " + "; ".join(d["connections"][:12]))
    if d.get("visible_text"):
        parts.append("Visible text: " + " ".join(str(t) for t in d["visible_text"][:40]))
    return " | ".join(p for p in parts if p)
