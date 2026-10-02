"""Thin Anthropic client with JSON extraction and an on-disk response cache.

The cache (keyed on model + system + content hash) makes evaluation re-runs free and
deterministic; delete data/cache/llm.sqlite to force fresh calls.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from ae import config
from ae.log import get_logger

log = get_logger(__name__)
CACHE = Path("data/cache/llm.sqlite")


class LLMError(Exception):
    pass


def _client():
    from anthropic import Anthropic

    key = config.api_key()
    if not key:
        raise LLMError("ANTHROPIC_API_KEY is not set (put it in .env)")
    return Anthropic(api_key=key)


def image_block(path: str | Path, max_side: int = 1500) -> dict:
    from io import BytesIO

    from PIL import Image

    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}}


def _cache_get(key: str) -> str | None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE)
    conn.execute("CREATE TABLE IF NOT EXISTS llm (k TEXT PRIMARY KEY, v TEXT)")
    row = conn.execute("SELECT v FROM llm WHERE k = ?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def _cache_put(key: str, value: str) -> None:
    conn = sqlite3.connect(CACHE)
    conn.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()


def complete(system: str, content: list[dict] | str, model: str | None = None, max_tokens: int = 1200, use_cache: bool = True) -> str:
    model = model or config.ANSWER_MODEL
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    key = hashlib.sha256(json.dumps([model, system, content], sort_keys=True).encode()).hexdigest()
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            log.debug(f"llm cache hit model={model}")
            return hit
    import time

    t0 = time.perf_counter()
    resp = _client().messages.create(model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": content}], timeout=120)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    usage = getattr(resp, "usage", None)
    log.debug(f"llm call model={model} {int((time.perf_counter() - t0) * 1000)} ms in={getattr(usage, 'input_tokens', '?')} out={getattr(usage, 'output_tokens', '?')} stop={getattr(resp, 'stop_reason', '?')}")
    if not text.strip():
        raise LLMError(f"empty model reply (stop_reason={getattr(resp, 'stop_reason', None)})")
    if use_cache:
        _cache_put(key, text)
    return text


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates ```json fences and prose)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError(f"no JSON object in model reply: {text[:200]!r}")
    s = m.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # trim to the last balanced brace
        depth = 0
        for i, ch in enumerate(s):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                return json.loads(s[: i + 1])
        raise


def complete_json(system: str, content: list[dict] | str, model: str | None = None, max_tokens: int = 1200) -> dict:
    text = complete(system, content, model, max_tokens)
    try:
        return extract_json(text)
    except Exception:
        # one retry asking for strict JSON
        text = complete(system + "\nReply with a single JSON object and nothing else.", content, model, max_tokens, use_cache=False)
        return extract_json(text)
