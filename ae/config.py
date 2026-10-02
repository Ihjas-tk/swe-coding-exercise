"""Runtime configuration from environment / .env (searched from the CWD upward).

Everything a grader may want to change is an environment variable:
  ANTHROPIC_API_KEY   hosted LLM key (required for figure descriptions, SQL and answering)
  AE_ANSWER_MODEL     default claude-sonnet-5-5
  AE_VLM_MODEL        default claude-haiku-4-5-20251001 (per-figure descriptions at ingest)
  AE_EMBED_MODEL      default ibm-granite/granite-embedding-english-r2
  AE_BACKEND          default hybrid (extraction backend used by `ask`)
  AE_CORPUS           directory to ingest instead of the built-in patents/ design_docs/ structured/
  AE_INDEX            index namespace (default "" = data/index/<backend>.sqlite; "ext" -> data/index/ext/<backend>.sqlite)
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv() -> None:
    for d in [Path.cwd(), *Path.cwd().parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


load_dotenv()

ANSWER_MODEL = os.environ.get("AE_ANSWER_MODEL", "claude-sonnet-5-5")
VLM_MODEL = os.environ.get("AE_VLM_MODEL", "claude-haiku-4-5-20251001")
EMBED_MODEL = os.environ.get("AE_EMBED_MODEL", "ibm-granite/granite-embedding-english-r2")
BACKEND = os.environ.get("AE_BACKEND", "hybrid")
CORPUS = os.environ.get("AE_CORPUS")  # None = built-in corpus dirs
INDEX = os.environ.get("AE_INDEX", "")


def api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")
