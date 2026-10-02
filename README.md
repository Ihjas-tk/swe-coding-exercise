# Answer engine for engineering documents

Ingests patents, design documents (PDF/DOCX), a bill of materials and a test log, and
answers natural-language questions with cited answers, or says "not found". Everything
except the LLM runs locally with open-source packages. The take-home brief is in
`Take-Home_Answer_Engine_for_Engineering_Documents.pdf` (summary in `docs/brief-summary.md`).

```
make setup                       # uv venv + pinned deps
make ingest                      # extract → structured tables → numeral index → chunks → index → embeddings → figure descriptions
make ask Q="What is the maximum discharge current rating of the EV-BMS-100?"
make eval                        # staged evaluation for all three extraction backends → data/eval/RESULTS.md
```

```json
{"answer": "The EV-BMS-100 max discharge current is 320 A continuous, 450 A for 10 s.",
 "citations": [{"doc": "EV-BMS-100_design_document.pdf", "page": 2}],
 "not_found": false}
```

## Setup

System packages: [uv](https://docs.astral.sh/uv/), Tesseract 5 (`brew install tesseract`),
LibreOffice (`brew install --cask libreoffice`; only needed for DOCX page numbers — without
it DOCX blocks are reported on page 1 and the output says so). Python 3.12 is pinned via
`.python-version`; `make setup` creates the environment from `uv.lock`.

Hosted model: put `ANTHROPIC_API_KEY=...` in a `.env` file (searched from the working
directory upward). Models are configurable: `AE_ANSWER_MODEL` (default `claude-sonnet-5-5`),
`AE_VLM_MODEL` (default `claude-haiku-4-5-20251001`, used once per figure at ingest),
`AE_EMBED_MODEL` (default `ibm-granite/granite-embedding-english-r2`, local), `AE_BACKEND`
(default `hybrid`). `make ingest` without a key skips the figure descriptions and still works.
First runs download the Docling layout/table models and the embedding model from Hugging Face.

`make ingest-all` builds the three indexes `make eval` compares; `make test` runs the unit
tests (structured loading, numeral index, chunker, routing). The EasyOCR ablation needs
`uv sync --group ablation` and `AE_DOCLING_OCR=easyocr`.

## Architecture

```
raw files
  PDF  ─► thin: PyMuPDF text/images + pdfplumber grids + Tesseract OCR (per page, when needed)
       ─► docling: layout model + TableFormer + OCR
       ─► hybrid: thin everywhere; Docling's tables/figures merged in on OCR'd pages   ◄── default
  DOCX ─► python-docx (exact tables, images, captions) + LibreOffice render for page numbers
  CSV/XLSX ─► typed SQLite tables (+ one text line per row for search)
            │ ParsedDocument: pages → ordered text / table / figure blocks, each with doc, page, bbox
            ▼
  numeral index ("control unit 160" from the specification text; figure defs; OCR label reconciliation)
  chunker: text (≤300 tok, one section, one page) · claim · table_row + table · figure (caption + labels + glossary + VLM description) · structured_row
            ▼
  SQLite: chunks · FTS5 (stemmed prose | exact identifiers | trigram) · embeddings · structured tables · numerals
            ▼
  question ─► parse (doc / figure / numeral / aggregate patterns)
           ├─ numeral route: index lookup                ─┐
           ├─ sql route: text-to-SQL, read-only, 1 retry ─┼─► evidence (short ids e1..en, mapped to doc/page by us)
           └─ hybrid retrieval (always): BM25 ∪ dense → RRF k=60 → page scoring → top pages + neighbours ─┘
            ▼
  Claude answers from evidence only (figure crop attached for visual questions) → JSON
            ▼
  verify: citations ⊆ supplied ids · named-document scope · every number in the answer appears in cited text · overlap warning
          any hard failure → not_found
```

Module map: `ae/extract/` (thin/, docling_backend.py, pagemap.py, structured.py, base.py),
`ae/index/` (numerals, chunker, identifiers, store, embed, vlm, ingest), `ae/retrieve/`
(query, hybrid, sql), `ae/answer/` (generate, verify, pipeline), `ae/eval/` (extraction,
omnidocbench, retrieval, answers, report), `gold/` (hand-annotated extraction gold set and
adversarial unanswerable questions), `docs/extraction.md` (how extraction works),
`docs/ai-tools.md`.

## Choices and why

| Choice | Why (evidence in `docs/extraction.md` and the research notes behind each module docstring) |
|---|---|
| Three extraction backends behind one schema | The brief grades understanding of the parser; `thin` is fully explainable, Docling is the strongest open layout model. Measuring both decided the default rather than guessing. |
| `hybrid` as default | On this corpus thin and Docling tie end-to-end; on scanned pages thin's text is cleaner (CER 0.13 % vs 21.8 %) while Docling finds tables and figures from pixels (TEDS 0.53 vs 0, 36/50 vs 0/50 figures). Hybrid takes each where it measured best. |
| PyMuPDF + pdfplumber + Tesseract | Exact glyphs, bboxes, ruled-table grids, column-aware OCR; each small enough to explain line by line. |
| Numeral index from text, OCR/VLM only confirm | Published figure-label OCR tops out around F1 0.71; patent specifications define every numeral. |
| Row + whole-table chunks, deterministic prefixes | Row boundaries and title prefixes have the only consistent measured gains in chunking studies; LLM-generated chunk context did not replicate reliably. |
| BM25 ∪ dense, RRF k=60, identifier column | Exact ids drive these questions; embedders are near chance on numbers. Dense adds paraphrase recall (it alone gets every table question). |
| granite-embedding-english-r2 | Apache licence, no prefixes, best table-retrieval score among base-size models; bge-small kept as the fast fallback (10× cheaper, ~1 question worse at R@3). |
| Claude Sonnet for answers/SQL, Haiku for figure descriptions | Image input needed for figures; descriptions are cached once per figure. |
| SQLite for everything | FTS5, blobs, structured tables and the numeral index in one file per backend; brute-force cosine is fine to ~100k chunks. |
| Rule router, hybrid always runs | A rule/TF-IDF router beat embedding classifiers in routing benchmarks; running retrieval alongside means a wrong route cannot fail silently. |
| Abstention = gates + deterministic checks | Models answer 40–99 % of unsupported questions when merely told to abstain; citation subset, document scope and number-grounding checks are what catch the rest. |

## Results

All numbers from `make eval` (`data/eval/RESULTS.md` has every breakdown and ablation).
The dev set has 20 answerable + 2 unanswerable questions; the adversarial set adds 30
unanswerable questions (`gold/unanswerable.json`); the extraction gold set is in `gold/`;
the external slice is 106 English double-column OmniDocBench pages (images, so all
backends run their OCR path there).

| Metric | thin | docling | hybrid |
|---|---|---|---|
| Extraction: OCR CER / WER (scanned patent page) | 0.0013 / 0.0080 | 0.2182 / 0.2122 | 0.0013 / 0.0080 |
| Extraction: table cell accuracy (4 tables, 117 cells) | 0.983 | 0.966 | 0.983 |
| Extraction: figures found / caption linked (9) | 9/9 / 9/9 | 9/9 / 3/9 | 9/9 / 9/9 |
| Extraction: figure label recall, Tesseract / VLM | 0.39 / 0.96 | 0.10 / 0.95 | 0.39 / 1.00 |
| Extraction: reading order pair accuracy (3 two-column pages) | 0.903 | 0.942 | 0.903 |
| OmniDocBench (106 scanned pages): text similarity | 0.749 | 0.837 | 0.733 |
| OmniDocBench: reading order pair accuracy | 0.948 | 0.979 | 0.940 |
| OmniDocBench: table TEDS (19 tables) | 0.00 | 0.534 | 0.534 |
| OmniDocBench: figure recall @ IoU 0.5 (50 figures) | 0.00 | 0.72 | 0.72 |
| Retrieval: page Recall@1 / @5, BM25 only | 0.60 / 0.90 | 0.60 / 0.95 | 0.60 / 0.90 |
| Retrieval: page Recall@1 / @5, dense only | 0.75 / 0.95 | 0.80 / 1.00 | 0.75 / 0.95 |
| Retrieval: page Recall@1 / @5, hybrid | 0.75 / 0.90 | 0.75 / 0.95 | 0.75 / 0.90 |
| Answers: correct (20 answerable; numbers within 1 % + LLM grader + read by hand) | 20/20 | 20/20 | 20/20 |
| Answers: cited page in gold evidence | 18/20 | 18/20 | 18/20 |
| Answers: citation precision | 0.775 | 0.775 | 0.775 |
| Answers: wrong declines on answerable | 0 | 0 | 0 |
| Not-found: handled correctly (2 dev + 30 adversarial) | 32/32 | 32/32 | 32/32 |
| Penalty score (+1 correct / 0 declined / −2 wrong) | 20 of 20 | 20 of 20 | 20 of 20 |

Retrieval by question type (hybrid backend, hybrid retrieval): factual 5/5, table 7/7,
figure 3/4, structured 4/4 at R@5; dense alone scores 0.25 on figure questions and BM25
alone 0.75, which is why figure questions also go through the numeral index. With 20
questions a difference of one question is 5 points; the backend differences in retrieval
are within that noise.

### Failure analysis

| Question | What went wrong | Stage |
|---|---|---|
| q02 "what component reduces the transfer head's traverse rate" | Correct answer; cited page 1 of the scanned patent, dev set cites page 2. The sentence and claim 1 are on page 1; page 2 holds claims 4+. | citation (gold label) |
| q07 "IP rating of the EV-BMS-100 enclosure" | Correct answer (IP67); cited page 2, dev set cites page 3. The mechanical table renders on page 2. | citation (gold label) |
| q12 "rated and peak torque of the RJA-40" (passes only via policy) | LibreOffice renders rows 1–4 of the spec table on page 1, the dev set cites page 2 (Word keeps the table with its heading). We cite every page a split table spans. | DOCX pagination |
| q16 "how many M12 ports in Figure 2" (passes only via policy) | The figure is on page 2, its caption on page 3; the dev set cites the caption's page. We cite both. | cross-page caption |
| Falcon-VT1 "Processor" row (extraction gold, 2 of 30 cells) | The PDF overprints "MHz" and "with" at the same coordinates; every parser splits it the same way. | extraction (corpus defect) |
| Figure numerals via Tesseract (recall 0.39) | Leader lines touching a "1" turn 100/110/120 into 00/10/20; the gripper's "160" lies outside the embedded image entirely. Repaired by reconciliation against the text where unique; the VLM description reads 0.96. | extraction |
| Docling captions (3 of 9) | Docling attaches the title it OCRs *inside* the picture as the caption; the real caption stays loose text. | extraction (docling) |
| Docling OCR on the scanned page (CER 0.22) | Docling emits the header-area blocks twice. Same with its default EasyOCR engine (CER 0.23, `AE_DOCLING_OCR=easyocr`), so it is overlapping layout regions, not the OCR driver. | extraction (docling) |
| Thin backend on OmniDocBench scans | No tables (needs ruling lines) and no figure boxes at IoU 0.5 (caption-driven boxes are approximate). | extraction (thin), fixed by hybrid |
| Citation precision 0.775 | Extra cited pages are the split-table / caption-page policy above and multi-page support (UAV battery answer draws on page 1 text and page 2 claim 3). | policy |

Assumptions written down because they differ from the dev set: the three gold-label
page differences above; `page` for CSV/XLSX is 1; a figure's citation includes its caption
page; a split table's citation includes every page it spans.

### External corpus (real documents)

To find gaps the synthetic corpus cannot show, 11 real files (two Google Patents grants, a
Microchip datasheet, a NASA paper, an arXiv paper, 12 ICDAR 2013 table pages, two scanned
NASA reports, a CubeSat DOCX procedure, an open-hardware BOM, the UCI SECOM test log) were
sourced with 52 hand-written questions (`gold/external_questions.json`, `make external`).
Final score on the default backend: 40/44 answerable correct, 42/44 cited pages, 8/8
unanswerable declined; the first pass scored 33/44, and the ten gaps fixed in between
(span-joined numbers, spreadsheet headers, multi-sheet ids, mixed-type columns, wide
schemas, totals rows, DOCX page mapping/headings/captions, verifier strictness) are listed
with the four still open in `docs/external-eval.md`.

## Known limitations

- Scanned tables are found only through Docling (hybrid/docling backends); thin has no pixel-based table detector.
- Thin's figure boxes on scans are caption-driven (one figure per caption, one column); uncaptioned drawings are invisible to it.
- The text-quality gate is a token heuristic; a plausible-looking but wrong text layer passes it.
- The numeral index is regex-based: lists with shared names ("left and right wings 14, 16") get the whole phrase; numerals reused across figures are not disambiguated by figure.
- DOCX pagination follows LibreOffice; Word may break pages differently.
- granite-r2 embeds ~50 ms per chunk on an M-series CPU (≈15 min for a 100-page document); set `AE_EMBED_MODEL=BAAI/bge-small-en-v1.5` for 10× faster ingest.
- Docling runs ~3 s/page on CPU (5–6 s on large scans); only OCR'd pages pay it in the hybrid backend.
- The eval is small: 20 answerable questions, so one question is 5 points; the adversarial set was written by us.
- No reranker, no page-image retrieval, no LLM-generated chunk context: all considered, measured as low-value for this corpus in the literature, left behind flags or out.
- From the external corpus: superscripts lost in OCR text layers; numbers that exist only inside figures; prose outranked by many short table rows in table-heavy documents (see `docs/external-eval.md`).

## With more time

1. Find why Docling duplicates header-area regions on the scanned patent (it is not the OCR engine: EasyOCR gives the same result) and whether a layout post-processing option fixes it; the EasyOCR run on the 106 OmniDocBench pages was not repeated.
2. Pixel-based figure finder for thin on scans (connected components of non-text ink).
3. Add distractor documents and re-measure Recall@k; tune the retrieval gate threshold on a larger unanswerable set with confidence intervals.
4. Cross-encoder reranker ablation (flag exists in the design, not wired).
5. Full-corpus-in-prompt baseline to keep retrieval numbers honest on small corpora.
6. Hidden-test readiness: a `--full` mode that attaches every figure of the named document for visual questions.

## AI coding tools

See `docs/ai-tools.md`.
