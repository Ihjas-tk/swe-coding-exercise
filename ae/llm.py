"""Minimal Anthropic client with JSON extraction and an on-disk response cache.

The cache (keyed on model + system + content hash) makes evaluation re-runs free and
deterministic; delete data/cache/llm.sqlite to force fresh calls.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ae import config
from ae.log import get_logger

log = get_logger(__name__)

CACHE = Path("data/cache/llm.sqlite")
"""Response cache keyed on model + system prompt + content."""
IMAGE_MAX_SIDE = 1500
"""Images are downscaled so their longest side is at most this many pixels (API size/cost limit)."""
REQUEST_TIMEOUT_S = 120
"""Seconds before one API request is abandoned (the SDK raises APITimeoutError)."""
DEFAULT_MAX_TOKENS = 1200
"""Reply budget when the caller sets none."""


class LLMError(Exception):
    """Missing key, empty reply or a reply without a JSON object."""


def _client() -> Any:
    """Return an `anthropic.Anthropic` client; raises LLMError when ANTHROPIC_API_KEY is unset.

    Typed Any on purpose: requests are built from plain dicts, not the SDK's TypedDicts.
    """
    # Deferred: the SDK import is slow and only needed when a call misses the cache.
    from anthropic import Anthropic

    key = config.api_key()
    if not key:
        raise LLMError("ANTHROPIC_API_KEY is not set (put it in .env)")
    return Anthropic(api_key=key)


def image_block(path: str | Path, max_side: int = IMAGE_MAX_SIDE) -> dict[str, Any]:
    """Return an Anthropic image content block for an image file (re-encoded as PNG, longest side capped)."""
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()},
    }


def _cache_get(key: str) -> str | None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(CACHE)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS llm (k TEXT PRIMARY KEY, v TEXT)")
        row = conn.execute("SELECT v FROM llm WHERE k = ?", (key,)).fetchone()
    return row[0] if row else None


def _cache_put(key: str, value: str) -> None:
    with closing(sqlite3.connect(CACHE)) as conn:
        conn.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, value))
        conn.commit()


def complete(
    system: str,
    content: list[dict[str, Any]] | str,
    model: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    use_cache: bool = True,
) -> str:
    """Return the reply text of one completion (cached unless `use_cache=False`).

    `model` defaults to config.ANSWER_MODEL. Raises LLMError when the key is missing or the
    reply is empty; SDK errors (timeouts, rate limits) propagate.
    """
    model = model or config.ANSWER_MODEL
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    key = hashlib.sha256(json.dumps([model, system, content], sort_keys=True).encode()).hexdigest()
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            log.debug(f"llm cache hit model={model}")
            return hit
    resp = _client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        timeout=REQUEST_TIMEOUT_S,
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    usage = getattr(resp, "usage", None)
    log.debug(
        f"llm call model={model} in={getattr(usage, 'input_tokens', '?')} out={getattr(usage, 'output_tokens', '?')} stop={getattr(resp, 'stop_reason', '?')}"
    )
    if not text.strip():
        raise LLMError(f"empty model reply (stop_reason={getattr(resp, 'stop_reason', None)})")
    if use_cache:
        _cache_put(key, text)
    return text


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model reply (tolerates ```json fences and prose).

    Raises LLMError when there is no object at all and json.JSONDecodeError when it is malformed.
    """
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError(f"no JSON object in model reply: {text[:200]!r}")
    s = m.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # The greedy match may run past the object into trailing prose: trim to the brace
        # that closes the first one.
        depth = 0
        for i, ch in enumerate(s):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                return json.loads(s[: i + 1])
        raise


def complete_json(
    system: str, content: list[dict[str, Any]] | str, model: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS
) -> dict[str, Any]:
    """Return a completion parsed as JSON, with one uncached retry that asks for strict JSON."""
    text = complete(system, content, model, max_tokens)
    try:
        return extract_json(text)
    except (LLMError, json.JSONDecodeError):
        text = complete(
            system + "\nReply with a single JSON object and nothing else.", content, model, max_tokens, use_cache=False
        )
        return extract_json(text)
