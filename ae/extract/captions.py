"""Caption patterns and figure-id normalisation shared by every extraction backend.

Native (PDF and DOCX) and Docling all derive `FigureBlock.figure_id` from a caption with
`figure_id()`, so "FIG. 2", "Fig 2" and "fig.2" index and cite as the same figure whichever
backend produced them. The canonical forms are the native backend's: "FIG. <n>" for the
FIG / Fig. spellings and "Figure <n>" for the spelled-out word (the two are kept apart
because the design documents and the patents number their figures independently).
"""

from __future__ import annotations

import re

CAPTION_RE = re.compile(r"^\s*(FIG(?:URE)?\.?|Figure|Table)\s*\d+[A-Za-z]?\s*([—–:\-]|\.(?=\s+[A-Z])|\.?\s*$)", re.I)
"""A caption line: "FIG. 2 —", "Figure 6. Text", "Table 1:", or a bare "FIG. 2"."""

FIG_TITLE_RE = re.compile(r"^\s*(FIG(?:URE)?\.?|Figure)\s*\d+[A-Za-z]?\.?\s*$", re.I)
"""A bare figure title ("FIG. 2") printed above a drawing, with no caption text."""

FIG_ID_RE = re.compile(r"^\s*(FIG(?:URE)?\.?|Figure)\s*(\d+[A-Za-z]?)", re.I)
"""The figure word and number at the start of a caption."""


def figure_id(caption: str | None) -> str | None:
    """Return the normalised figure id of a caption ("FIG. 2" / "Figure 1"), or None.

    "FIG", "FIG.", "Fig." in any case become "FIG."; "Figure"/"FIGURE" become "Figure".
    The number keeps its letter suffix, upper-cased ("FIG. 3A").
    """
    if not caption:
        return None
    m = FIG_ID_RE.match(caption)
    if not m:
        return None
    word = m.group(1).upper()
    label = "FIG." if word.startswith("FIG.") or word == "FIG" else "Figure"
    return f"{label} {m.group(2).upper()}"
