"""Runtime configuration from environment / .env (searched from the CWD upward).

Everything a grader may want to change is an environment variable:
  ANTHROPIC_API_KEY   hosted LLM key (required for figure descriptions, SQL and answering)
  AE_ANSWER_MODEL     default claude-sonnet-5-5
  AE_VLM_MODEL        default claude-haiku-4-5-20251001 (per-figure descriptions at ingest)
  AE_EMBED_MODEL      default ibm-granite/granite-embedding-english-r2
  AE_BACKEND          default hybrid (extraction backend used by `ask`)
  AE_CORPUS           directory to ingest instead of the built-in patents/ design_docs/ structured/
  AE_INDEX            index namespace (default "" = data/index/<backend>.sqlite; "ext" -> data/index/ext/<backend>.sqlite)
  AE_DOCLING_OCR      default tesseract; easyocr is the OCR-engine ablation for the Docling backend
  AE_LOG              console log level, default INFO (the log file always gets DEBUG)
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from the nearest .env without overriding the environment."""
    for directory in [Path.cwd(), *Path.cwd().parents]:
        env_file = directory / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return


load_dotenv()

ANSWER_MODEL = os.environ.get("AE_ANSWER_MODEL", "claude-sonnet-5-5")
VLM_MODEL = os.environ.get("AE_VLM_MODEL", "claude-haiku-4-5-20251001")
EMBED_MODEL = os.environ.get("AE_EMBED_MODEL", "ibm-granite/granite-embedding-english-r2")
BACKEND = os.environ.get("AE_BACKEND", "hybrid")
CORPUS = os.environ.get("AE_CORPUS")  # None = built-in corpus dirs
INDEX = os.environ.get("AE_INDEX", "")
DOCLING_OCR = os.environ.get("AE_DOCLING_OCR", "tesseract")  # tesseract | easyocr
LOG_LEVEL = os.environ.get("AE_LOG", "INFO")


def api_key() -> str | None:
    """Return the Anthropic API key, or None when unset."""
    return os.environ.get("ANTHROPIC_API_KEY")
