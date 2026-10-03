# Note on AI coding tools

## What I used

Claude Code, running Claude Fable 5.1, for the whole project. For research and for some of the
larger work packages it ran Claude Opus subagents in parallel. No other coding assistants,
no AI-generated code copied in from elsewhere. The answering pipeline itself calls the
Anthropic API (Sonnet for answers, Haiku for figure descriptions and grading); that is part of
the product, not of how it was built.

## How the work was split

I treated the assistant as a fast pair programmer and kept the decisions. My side of it:

- Set the direction at each phase (PDF extraction first, then DOCX, structured files, indexing,
  answering, evaluation) and reviewed each design in conversation before any code was written.
- Made the calls that shaped the system: comparing a hand-built extraction stack against
  Docling before choosing, keeping Tesseract as the only OCR engine, the Anthropic API and the
  model tiers, installing LibreOffice for DOCX page numbers, choosing OmniDocBench and a set of
  real documents as external evaluation data, citing every page a split table spans, and
  removing the losing alternatives from the code once the comparison was done.
- Reviewed the external evaluation questions and the gold data before they were used, and
  reviewed every results table before it went into the README.

The assistant's side: literature research before each module (chunking, retrieval, embedding
models, figure question answering and abstention, each as a sourced findings file), writing
the code and tests, running them on the corpus after every change and reporting what broke,
drafting the documentation, and sourcing the external documents.

## How I checked its work

- **Measurement over opinion.** Every "X is better than Y" in the README is backed by a number
  from the evaluation harness, and the harness was built before the comparisons were run.
- **Claims about libraries were checked against their source.** For example how Docling
  orders blocks (its `reading_order_rb.py`) and how pdfplumber snaps table edges, before
  either was described in `docs/extraction.md`.
- **Mistakes were caught and corrected.** The assistant once reported that Docling had
  dropped four lines of text; checking its Markdown export showed the lines were merged, not
  lost, and the claim was withdrawn. It mis-transcribed one external-corpus reference answer
  (a timestamp ordering) and a few figure labels in the gold set, found when the evaluation
  disagreed with the gold and the page image was re-read. Its first evaluation run also counted
  stale figure descriptions from an earlier run, which inflated one metric until the ingest
  was made to clear them.
- **Clean-room runs.** The final code was set up and run three times from a fresh copy with
  no caches and no downloaded models, to make sure the one-command setup and the numbers hold
  outside the machine state it was developed in.

