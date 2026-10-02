# Take-Home: Answer Engine for Engineering Documents

The full exercise brief is included as `Take-Home_Answer_Engine_for_Engineering_Documents.pdf`.
This README summarizes it and tells you exactly what's in this package.

## Context

Hardware engineering teams in robotics, aerospace, EV, and similar fields work from a messy mix
of documents: patents, design documents full of block diagrams and spec tables, bills of
materials, and test logs. When an engineer asks a real question, the answer is rarely in one
clean paragraph. It may sit in a table on page 14, in a figure label, in a scanned patent page,
or in a spreadsheet row.

Your task is to build an **answer engine** that ingests such a corpus and answers
natural-language questions with **grounded, cited answers**. Just as importantly, we want you to
**show how good it is** through an evaluation you design and run.

We've kept the scope deliberately focused. We'd rather see a small system that you understand end
to end, measure honestly, and can reason about than a broad one held together by black boxes.
**Please submit within 7 days.**

## What we provide (this package)

```
Take-Home_Answer_Engine_for_Engineering_Documents.pdf   <- the full brief (read this first)

patents/                 4 patents (PDF)
  US2988237A_programmed_article_transfer.pdf     <- scanned, image-only (no text layer) — requires OCR
  US8485576B2_robotic_gripper.pdf                 <- two-column layout, numbered claims
  US5481460A_ev_controller.pdf                    <- two-column layout, numbered claims
  US20170320570A1_uav_vtol.pdf                    <- two-column layout, numbered claims

design_docs/             3 design documents (PDF/DOCX)
  EV-BMS-100_design_document.pdf                  <- block diagram, spec tables, enclosure figure
  Falcon-VT1_flight_controller_design_document.pdf <- block diagram, spec table
  RJA-40_actuator_design_document.docx             <- block diagram, spec table, assembly figure

structured/               2 structured files
  test_log.csv            <- hardware test log
  bill_of_materials.xlsx  <- bill of materials

dev_set/
  questions.json          <- ~20 questions with reference answers and evidence location
                             (document + page), spanning factual lookup, table lookup,
                             figure-grounded, structured-data, and unanswerable questions.
                             Use this to build and tune your system.
```

A separate **hidden test set** of similar questions will be used to evaluate your system after
submission; it is not included here.

Patents use two-column layouts, numbered claims, and figures whose parts are labelled with
reference numerals (e.g. "valve 112" in the text maps to the label 112 in FIG. 3). Design
documents contain block diagrams, photos or CAD screenshots, and spec tables (electrical ratings,
dimensions, operating limits).

## Constraints

**1. Models are your choice.** Any LLM, VLM, or embedding model, open-weight or hosted (via an API
key). Tell us what you used and why. If you use a hosted model, make it configurable so we can
plug in our own key.

**2. Everything else must be open source and run locally.** Parsing, OCR, layout detection, table
extraction, image extraction, chunking, indexing, and retrieval must be done with open-source
packages running in your own environment.
- Not allowed: hosted document-processing or RAG services (LlamaParse, Unstructured's hosted API,
  AWS Textract, Azure Document Intelligence, Google Document AI, Reducto, managed vector databases).
- Allowed: open-source libraries and the models bundled with them (e.g. Docling, Marker, PyMuPDF,
  pdfplumber, Camelot, Tesseract, PaddleOCR, Surya, Table Transformer) and self-hosted storage
  (e.g. Qdrant, pgvector, LanceDB, DuckDB, SQLite).
- Grey area, clarified: you may call a general-purpose LLM/VLM inside your pipeline (e.g. to
  describe a figure or interpret a table), but the pipeline itself — what gets extracted, how it's
  structured, how it links back to the source page — must be yours.

**3. Know your tools from the inside.** For each extraction library you rely on, be able to
explain how it detects page layout, decides reading order (especially two-column pages), recovers
table structure, locates and crops figures, when OCR kicks in and which engine it uses, and where
it breaks (ideally with an example from this corpus). We'll spend a good part of the follow-up
discussion on this.

**4. Easy for us to run.** We should be able to clone your repo and run ingestion, querying, and
evaluation with one command each (e.g. `make ingest`, `make ask Q="..."`, `make eval`). Pin your
dependencies and document any system packages (e.g. Tesseract).

## Requirements

**Ingestion and extraction** — text in correct reading order (two-column patent pages should not
interleave line by line; scanned pages go through OCR); tables with row/column structure intact;
figures extracted, linked to their caption, and made findable; the CSV/XLSX ingested so they can
be queried precisely (e.g. with SQL); every extracted piece carries its source document and page.

**Retrieval and answering** — any retrieval approach (keyword, vector, hybrid, page-image based,
SQL, or a mix), explained briefly; every answer cites document and page; when the corpus doesn't
contain the answer, say "not found" instead of guessing; a CLI or minimal HTTP API is enough; make
output machine-readable for automatic scoring, in this shape:

```json
{
  "answer": "The rated continuous current is 120 A.",
  "citations": [{"doc": "motor_controller_design.pdf", "page": 7}],
  "not_found": false
}
```

**Question types** your system should handle: factual lookup from text, table lookup,
figure-grounded, structured data (CSV/XLSX), and unanswerable questions.

## Evaluation

Build a harness that runs with one command and measures quality **at each stage**, so when an
answer is wrong you can tell whether extraction, retrieval, or generation caused it.

- **Extraction quality** — you can't measure this against the dev set alone, so create a small
  gold set yourself: hand-annotate a handful of pages covering a couple of tables, a figure, and
  the scanned page. Report metrics you can defend (OCR character/word error rate, cell-level table
  accuracy, figure-to-caption linking).
- **Retrieval quality** — Recall@k against the gold evidence pages in the dev set, broken down by
  question type.
- **Answer quality** — correctness (with a stated tolerance for numbers; explain how you judged
  free-text answers), citation accuracy, and "not found" behavior (how often it correctly declines
  vs. wrongly declines on answerable ones).
- **Failure analysis** — a short table of failures, each with the question, what went wrong, and
  which stage caused it. Summarize all results in one table in your README.

## Deliverables

1. **Git repository** with code and one-command setup for ingestion, querying, and evaluation.
2. **README** covering architecture overview, model/library choices and why, evaluation results
   table and failure analysis, known limitations, and what you'd do next with more time.
3. **"How extraction works"** (roughly a page, in the repo): for text, tables, images, and OCR,
   what your chosen library does under the hood, why you picked it over alternatives, and one case
   where it fails on this corpus and what you did or would do about it.
4. **A short note on AI coding tools** — using them is allowed and expected; tell us how you used
   them.

## How we'll assess

Rigor and honesty of the evaluation; hidden test set results (especially tables, figures, and
correct "not found" answers); your understanding of your extraction stack; engineering quality;
and judgment about what you chose not to build.

## Follow-up discussion

A 60–90 minute session where you walk us through your repo, results, and failure analysis; we go
deep on how your parser, table extraction, OCR, and image extraction work internally; and we add a
new document or question type together to see how your system and evaluation hold up.

## Questions

Email us if anything is unclear. If you'd rather not wait, make a reasonable assumption and write
it down in your README; that's completely fine.
