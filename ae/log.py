"""Logging for every stage.

`get_logger(name)` returns a logger under the "ae" hierarchy that writes INFO+ to stderr
(level from AE_LOG via ae.config, default INFO) and DEBUG+ to data/logs/ae.log (rotating,
5 MB x 3). Conventions: INFO per-document milestones, DEBUG per-page detail, WARNING for
degraded modes (no LibreOffice, OCR fallback, VLM skipped).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ae import config

LOG_DIR = Path("data/logs")
"""Where the rotating DEBUG log file is written."""
LOG_FILE_MAX_BYTES = 5_000_000
"""Rotate the log file at ~5 MB ..."""
LOG_FILE_BACKUPS = 3
"""... keeping this many old files."""
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_NOISY = ("httpx", "urllib3", "sentence_transformers", "transformers", "docling", "torch", "PIL", "pdfminer")
"""Third-party loggers capped at WARNING (they log per request / per page at INFO)."""
_configured = False


def _configure() -> None:
    """Attach the console and file handlers to the "ae" logger, once per process."""
    global _configured
    if _configured:
        return
    _configured = True
    root = logging.getLogger("ae")
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    console.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
    root.addHandler(console)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_DIR / "ae.log", maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUPS
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)
    except OSError:
        pass  # read-only checkout: console logging only
    for noisy in _NOISY:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger; names outside the "ae" package are nested under it."""
    _configure()
    return logging.getLogger(name if name.startswith("ae") else f"ae.{name}")
