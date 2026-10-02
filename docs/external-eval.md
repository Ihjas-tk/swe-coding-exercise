# External evaluation: real documents the dev corpus does not cover

Purpose: find gaps in the pipeline by running it on documents the system was not built
against. Eleven files were sourced to cover each module (`gold/external_manifest.json`
has URLs, licences and checksums; `make external` re-fetches, ingests and scores them).
The 52 questions in `gold/external_questions.json` were written from the page images and
source files, not from parser output; answers were verified against the pages or
computed from the files. `data/eval/EXTERNAL.md` is the generated report.

| Module tested | Document | Why it was chosen |
|---|---|---|
| Patents (numerals, OCR text layer) | US9577240B2 battery module; US8485576B2 robotic gripper (the real version of the dev-corpus one) | suffixed/primed numerals reused across figures; prior-art 1020 vs 120; Google Patents' OCR text layer with interleaved columns and line numbers |
| Datasheet tables | Microchip MCP73831 | a table continuing across three pages, merged headers, notes in cells |
| Borderless tables, rotated headers | NASA X-57 power system paper | fully borderless Table I, rotated column headers in Table II |
| Figure/caption linking | arXiv 2106.01110 | full-width and multi-panel figures, captions below tables |
| Table cell ground truth | ICDAR 2013, 12 short documents | real PDFs with cell annotations, ruled and borderless |
| Text-quality gate, per-page routing | NASA 1965 telemetry report | clean text layer on some pages, garbage on others, image-only pages |
| Scanned ruled table | NASA TN D-3411 (1966) | two-level header on a scan; letter-spaced text layer |
| DOCX | BIRDS-5 thermal vacuum test procedure | 21 tables (merged cells, four across page breaks), 17 captioned figures, no heading styles |
| XLSX BOM | OLSK 3D printer BOM | four sheets, header on row 6, unnamed cost column, totals row, text in a numeric column |
| CSV log | UCI SECOM | 592 columns, pass/fail encoded as −1/1, not in chronological order |

## Results (final run)

| Metric | thin | docling | hybrid (default) |
|---|---|---|---|
| Correct, 44 answerable (numbers within 1 % and LLM grader) | 40 | 38 | 40 |
| Cited page in gold evidence | 42 | 38 | 42 |
| Wrong declines | 2 | 2 | 2 |
| Unanswerable handled correctly (8) | 8 | 8 | 8 |
| By type: factual / table / figure / structured | 12/14 · 15/15 · 7/9 · 6/6 | 12/14 · 14/15 · 6/9 · 6/6 | 12/14 · 15/15 · 7/9 · 6/6 |

Before the gaps below were fixed the same set scored 33 correct / 37 cited pages on the
default backend; the dev corpus stayed at 20/20 throughout (re-checked after every fix).

## Gaps found and fixed

| Gap | Trigger | Stage | Fix |
|---|---|---|---|
| Numbers split across font spans came out as "0 .45" | arXiv Table II | thin text extraction | spans joined by physical gap, not a blanket space |
| Spreadsheet header below title rows; a column with no header | OLSK BOM | structured loader | header = row with most text cells; unnamed columns `col_N` |
| Multi-sheet workbook rows collided on chunk ids | OLSK BOM | indexing | ids keyed on sheet |
| Numeric column with a few text cells typed TEXT, so MAX failed | OLSK cost column ("16H") | structured loader | ≥ 90 % numeric ⇒ numeric; minority cells NULL |
| 592-column schema prompt returned an empty model reply | SECOM | SQL route | wide schemas summarised; SQL failures fall back to retrieval |
| MAX() picked the totals row with a NULL label | OLSK BOM | SQL prompt | prompt tells the model to exclude NULL-label rows for per-item questions |
| DOCX table rows located by their first cell (often a row number) ⇒ page 1 | BIRDS-5 | DOCX page mapping | locate by the longest cell, inherit page otherwise |
| DOCX headings as bold numbered paragraphs, captions as "Figure 6." | BIRDS-5 | DOCX extraction | heading and caption fallbacks |
| Verifier rejected "Table 7" (label) and "0 V" (OCR'd as "OV") | BIRDS-5, battery patent | verification | label numbers ignored; OCR confusions tolerated |
| One figure-description timeout aborted the whole ingest | large patent sheet | VLM stage | retry once, then skip |
| Grader answered in prose instead of JSON | several | eval harness | one-word verdict fallback |

## Gaps left open (limitations)

- **Superscripts in OCR text layers**: "576 × 10⁸" reads "576 X 108"; the answer reproduces the corrupted text (x27).
- **Numbers read off figures**: the flat-cable width in the X-57 paper is only in the drawing; thin declines, the VLM reads 0.83 in for 0.61 in (x19).
- **Prose outranked by many short table rows**: the BIRDS-5 soak-time sentence ranks ~20th behind "Cold Soak 1 Start" rows (x39). Docling finds it (its chunking differs); a kind-aware prior or a reranker is the likely fix.
- **"Which figure shows X" on real patents**: the gripper's control unit appears in FIG. 11; thin picked the wrong sheet from the VLM descriptions (x04).
- **Docling-specific**: DOCX superscripts dropped ("1 x 10 Pa"), DOCX table pages mis-mapped, numeral definition pages differ from the gold (x02/x08/x36–x39).
- **Gold-set lessons**: two of my own expected answers were wrong or ambiguous (file order vs chronological order; cross-referenced numerals in references). Reading the page is not enough; compute what can be computed and keep references to the asked value.
