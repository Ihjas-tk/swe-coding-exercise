# Answer engine for engineering documents

Ingests patents, design documents (PDF/DOCX), a bill of materials and a test log, and
answers natural-language questions with cited answers, or says "not found". Everything
except the LLM runs locally with open-source packages. The take-home brief is in
`Take-Home_Answer_Engine_for_Engineering_Documents.pdf` (summary in `docs/brief-summary.md`).
It is a command-line tool (`ae`, wrapped by `make` targets); there is no HTTP server.

## Quick start

macOS (Homebrew) or Debian/Ubuntu (apt); everything else is installed by `setup.sh`.

```bash
git clone https://github.com/Ihjas-tk/swe-coding-exercise.git && cd swe-coding-exercise
./setup.sh            # installs what is missing, builds .venv from uv.lock, creates .env, downloads models, runs `ae check`
#   setup.sh offers to paste the key (hidden input); otherwise edit .env:  ANTHROPIC_API_KEY=sk-ant-...
make smoke            # ~6 min: one file of each type -> ingest, one question, 9 dev questions scored
make ingest-all       # ~3 min: the three full-corpus indexes (native, docling, hybrid)
make ask Q="What is the maximum discharge current rating of the EV-BMS-100?"
make eval             # ~6 min: staged evaluation of all three backends -> data/eval/RESULTS.md
```

`make ask` prints:

```json
{"answer": "The EV-BMS-100 max discharge current is 320 A continuous, 450 A for 10 s.",
 "citations": [{"doc": "EV-BMS-100_design_document.pdf", "page": 2}],
 "not_found": false}
```

`make ask` needs the index of its backend (`hybrid` by default): `make ingest` builds just that
one, `make ingest-all` builds all three. `make smoke` builds its own small index (`data/index/smoke/`)
and does not replace them. `make check` (also the last step of `setup.sh`) prints one line per
prerequisite; `!` lines are warnings, `✗` lines must be fixed before ingesting.

### Measured expectations

From a clean-room run (fresh clone, M-series MacBook, 16 GB, CPU only, Homebrew tools already present):

| Step | Time |
|---|---|
| `./setup.sh` (deps + models; add a few minutes when it has to install Tesseract or LibreOffice) | ~1 min |
| `make smoke` (includes first-time Docling and embedding work on 6 files) | ~6 min |
| `make ingest-all` (after models are cached) | ~3 min |
| `make eval` (all three backends; LLM replies are cached in `data/cache/llm.sqlite`, so re-runs are faster) | ~6 min |
| First question in a fresh process (loads the embedding model) | ~15 s |
| Each further question in the same process (e.g. inside `make eval`) | ~2–3 s |

Every `make ask` is a new process, so each one pays the ~15 s model load.

## What gets installed and downloaded

`setup.sh` checks before it installs, so re-running it is safe. Nothing is installed with `sudo`
except apt packages (and only when not already root).

| Item | Size | Where it lands | Needed for |
|---|---|---|---|
| [uv](https://docs.astral.sh/uv/) (official installer, if missing) | ~40 MB | `~/.local/bin/uv` | everything (Python 3.12 is fetched by uv into `~/.local/share/uv/python/` if absent) |
| Tesseract 5 + English data | ~40 MB | Homebrew prefix (`brew install tesseract`) / `/usr/bin` (`apt-get install tesseract-ocr tesseract-ocr-eng`) | OCR of scanned pages and figure labels (required) |
| LibreOffice (optional) | ~700 MB cask / ~300 MB apt | `/Applications/LibreOffice.app` (`brew install --cask libreoffice`) / `apt-get install libreoffice-writer` | DOCX page numbers; without it DOCX blocks are cited as page 1. Skip with `SKIP_LIBREOFFICE=1 ./setup.sh` |
| Python dependencies (pinned in `uv.lock`, incl. torch, Docling) | ~1.3 GB | `.venv/` in the repo | everything |
| Embedding model `ibm-granite/granite-embedding-english-r2` | ~290 MB | Hugging Face cache* | dense retrieval |
| Docling layout model (heron) + TableFormer | ~500 MB | Hugging Face cache* | `docling` and `hybrid` backends |
| Fallback embedding model `BAAI/bge-small-en-v1.5` | ~130 MB | Hugging Face cache* | `EMBED=BAAI/bge-small-en-v1.5` (10x faster ingest) |
| `.env` | — | repo root (git-ignored), from `.env.example` | `ANTHROPIC_API_KEY` and optional `AE_*` settings |
| Caches, indexes, reports, logs | grows to a few hundred MB | `data/` (`cache/`, `index/`, `eval/`, `logs/`) | created by the commands below |

\* `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`, the same resolution
`huggingface_hub` uses. `make models` (or `SKIP_MODELS=1 ./setup.sh` to defer) fetches them ahead of
the first ingest; `make check` reports whether they are cached.

## Configuration

`ANTHROPIC_API_KEY` goes in `.env` (read from the working directory, then each parent; only
the nearest `.env` is used; real environment variables win). `.env.example` documents the
optional settings: `AE_ANSWER_MODEL` (default `claude-sonnet-5-5`), `AE_VLM_MODEL` (default
`claude-haiku-4-5-20251001`, once per figure at ingest), `AE_EMBED_MODEL` (default
`ibm-granite/granite-embedding-english-r2`, local), `AE_BACKEND` (default `hybrid`),
`AE_CORPUS`, `AE_INDEX`, `AE_LOG`, `AE_DOCLING_OCR`, and the Hugging Face cache variables.
Without a key, `make ingest` still builds the index (figure descriptions are skipped);
answering needs it. Make variables: `BACKEND=native|docling|hybrid` (default `hybrid`),
`EMBED=<model id>` (default granite-r2; empty skips embeddings), `VLM=0|1` (figure
descriptions at ingest, default 1), `Q="..."` and `DEBUG=1` for `make ask`.

## Commands

`make help` prints this list from the Makefile.

| Target | What it does |
|---|---|
| `make setup` | runs `./setup.sh`: uv, Tesseract, LibreOffice, pinned deps, `.env`, models, `ae check` (re-runnable) |
| `make models` | pre-download the embedding model (`EMBED`, + bge-small fallback) and the Docling models into the HF cache |
| `make check` | verify Tesseract, LibreOffice, Python packages, `uv.lock`, `.env` + API key, cached models, indexes |
| `make smoke` | ~6 min end-to-end check on one file of each type (ingest + ask + score 9 dev questions) |
| `make extract` | extraction only; dumps `data/extracted/$(BACKEND)/` (JSON + Markdown per document) |
| `make structured` | load CSV/XLSX into SQLite and print the schema the LLM sees |
| `make ingest` | extract + structured + numerals + chunk + index + embeddings (+ figure descriptions when `VLM=1`) |
| `make ingest-nodense` | same, without embeddings (no embedding model needed) |
| `make ingest-all` | build the three indexes (`native`, `docling`, `hybrid`) used by `make eval` |
| `make ask Q="..."` | answer one question (`DEBUG=1` adds routing/retrieval/verification details) |
| `make eval` | full staged evaluation for all backends -> `data/eval/RESULTS.md` (builds any missing index first) |
| `make eval-fast` | extraction + retrieval stages only (no LLM calls) |
| `make external` | fetch the external evaluation corpus, ingest it with all backends and score it -> `data/eval/EXTERNAL.md` |
| `make test` | unit tests (routing tests need `make ingest BACKEND=native`) |
| `make lint` / `make fmt` | ruff format check + lint + mypy / apply the formatter and safe lint fixes |
| `make clean-cache` | delete page caches and extraction dumps (indexes are kept) |

Lower-level commands are in `uv run ae --help` (e.g. `ae search`, `ae show`, `ae eval --quick`).
The EasyOCR ablation needs `uv sync --group ablation` and `AE_DOCLING_OCR=easyocr`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `make ask` returns `"not_found": true` for everything; log says `generation failed: LLMError: ANTHROPIC_API_KEY is not set` | No key. Put `ANTHROPIC_API_KEY=...` in `.env` (`make check` shows which `.env` is read). A `.env` in the repo hides one in a parent directory. |
| `✗ tesseract not found`, or `... 'eng' language file is missing` | macOS: `brew install tesseract`. Debian/Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`. `tesseract --list-langs` must list `eng`. |
| `! libreoffice not found`; DOCX citations all say page 1 | Optional. `brew install --cask libreoffice` / `sudo apt-get install libreoffice-writer`, then rebuild the index (`make ingest` or `make ingest-all`). |
| Homebrew step is very slow or hangs | `HOMEBREW_NO_AUTO_UPDATE=1 ./setup.sh` skips the index update; a mirror can be set with `HOMEBREW_API_DOMAIN` / `HOMEBREW_BOTTLE_DOMAIN`; `SKIP_LIBREOFFICE=1 ./setup.sh` skips the largest download. |
| Model download fails (proxy, firewall, huggingface.co blocked) | Set `HF_ENDPOINT` to a reachable mirror and run `make models`; or copy `~/.cache/huggingface/hub` from another machine and set `HF_HUB_OFFLINE=1`. `BACKEND=native` needs no Docling models; `make ingest-nodense` needs no embedding model. |
| `index data/index/<backend>.sqlite not found; build it with make ingest BACKEND=<backend>` | `make ask` uses the full-corpus index of `BACKEND` (default `hybrid`); `make smoke` does not build it. Run the command in the message, or `make ingest-all`. |
| API timeouts / overloaded errors | Each call has a 120 s timeout and the SDK retries twice. A failed answer comes back as `not_found` with the error in `DEBUG=1` output; a failed figure description is retried once, then skipped. Successful replies are cached, so re-running `make eval` only re-sends the failures. |
| `! uv.lock ... run uv sync` in `make check` | The environment drifted from the lock file; `uv sync` restores it (extra packages such as the ablation group are allowed). |
| `uv: command not found` right after setup | The installer put uv in `~/.local/bin`; open a new shell or `export PATH="$HOME/.local/bin:$PATH"`. |

Logs: the console shows INFO (`AE_LOG=DEBUG` for more); the full DEBUG log is in `data/logs/ae.log`.

## Architecture

```
raw files
  PDF  ─► native: PyMuPDF text/images + pdfplumber grids + Tesseract OCR (per page, when needed)
       ─► docling: layout model + TableFormer + OCR
       ─► hybrid: native everywhere; Docling's tables/figures merged in on OCR'd pages   ◄── default
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

Module map: `ae/cli.py` (the `ae` command), `ae/config.py` (environment / `.env`), `ae/check.py`
(`make check`), `ae/models.py` (`make models`), `ae/smoke.py`, `ae/llm.py` (cached Anthropic calls),
`ae/schema.py` (ParsedDocument); `ae/extract/` (`native/` = pdf, text, tables, figures, ocr, docx;
docling_backend.py, pagemap.py, structured.py, captions.py, cache.py, base.py), `ae/index/` (numerals,
chunker, identifiers, store, embed, vlm, ingest), `ae/retrieve/` (query, hybrid, sql), `ae/answer/`
(generate, verify, pipeline), `ae/eval/` (extraction, omnidocbench, retrieval, answers, devset,
external, scoring, report), `gold/` (hand-annotated extraction gold set, adversarial unanswerable
questions, external question set), `docs/extraction.md` (how extraction works),
`docs/external-eval.md`, `docs/ai-tools.md`.

## Choices and why

| Choice | Why (evidence in `docs/extraction.md` and the research notes behind each module docstring) |
|---|---|
| Three extraction backends behind one schema | The brief grades understanding of the parser; `native` is fully explainable, Docling is the strongest open layout model. Measuring both decided the default rather than guessing. |
| `hybrid` as default | On this corpus native and Docling tie end-to-end; on scanned pages native's text is cleaner (CER 0.13 % vs 21.8 %) while Docling finds tables and figures from pixels (TEDS 0.53 vs 0, 36/50 vs 0/50 figures). Hybrid takes each where it measured best. |
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

| Metric | native | docling | hybrid |
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
| Native backend on OmniDocBench scans | No tables (needs ruling lines) and no figure boxes at IoU 0.5 (caption-driven boxes are approximate). | extraction (native), fixed by hybrid |
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

- Scanned tables are found only through Docling (hybrid/docling backends); native has no pixel-based table detector.
- Native's figure boxes on scans are caption-driven (one figure per caption, one column); uncaptioned drawings are invisible to it.
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
2. Pixel-based figure finder for native on scans (connected components of non-text ink).
3. Add distractor documents and re-measure Recall@k; tune the retrieval gate threshold on a larger unanswerable set with confidence intervals.
4. Cross-encoder reranker ablation (flag exists in the design, not wired).
5. Full-corpus-in-prompt baseline to keep retrieval numbers honest on small corpora.
6. Hidden-test readiness: a `--full` mode that attaches every figure of the named document for visual questions.

## AI coding tools

See `docs/ai-tools.md`.
