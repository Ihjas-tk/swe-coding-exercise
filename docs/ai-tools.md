# Note on AI coding tools

This repository was built with Claude Code (Claude Fable 5.1) driving the work in
conversation with the author, who set the direction at each step, reviewed outputs,
and made the design decisions recorded below.

How it was used:

- **Research before building.** Four background research tasks (run on Claude Opus)
  surveyed published evidence on chunking, retrieval strategy, embedding models, and
  figure QA / abstention, each producing a sourced findings file. Their recommendations
  (deterministic chunk prefixes, hybrid BM25 + dense with RRF, granite-embedding-r2,
  numeral index from text rather than OCR, stacked abstention checks) shaped modules
  1–10 and are cited in the module docstrings and README.
- **Writing code and tests.** The assistant wrote the modules phase by phase (PDF
  extraction, DOCX, structured, chunking, index, retrieval, answering, evaluation), ran
  them on the corpus after every change, and reported what broke. Several bugs were
  found that way and fixed (column clipping losing figure labels, identifier boost
  flooding BM25, a final block sort that undid column ordering, Docling items missing
  claim markers).
- **Gold set.** The scanned-page transcription, table cells, figure labels and
  reading-order anchors in `gold/` were produced by the assistant reading the page
  images, then checked by the author. The adversarial unanswerable questions were
  written by the assistant and checked against the corpus.
- **Verification habit.** Claims about library behaviour were checked against the
  installed source (e.g. Docling's `reading_order_rb.py`, pdfplumber's edge snapping)
  and against measurements on this corpus before being written into
  `docs/extraction.md`. One early claim (that Docling dropped four lines of text) was
  retracted after checking its markdown export: the lines were merged, not lost.

What the assistant did not decide: the backend comparison plan, the choice of the
Anthropic API and model tiers, installing LibreOffice, the external benchmark choice,
and the policy of citing every page a split table spans were all the author's calls.
