# How extraction works

Three backends produce the same `ParsedDocument` (pages → ordered text / table / figure
blocks, each with document, page and bounding box). `native` is a hand-assembled stack of
small libraries; `docling` is IBM's layout-model pipeline; `hybrid` uses `native` for every page and, on pages
that needed OCR, adds Docling's table and figure blocks (native's text, Docling's layout:
each where it measured best).

## Text and reading order — PyMuPDF (`ae/extract/native/text.py`)

**What the library does.** `page.get_text("dict")` walks the page's content stream and
groups glyphs into spans → lines → blocks using MuPDF's structured-text device: glyphs on
one baseline become a line, lines with similar font and small vertical gaps become a
block. Blocks come back in *content-stream order*, i.e. the order the PDF writer emitted
them, which is usually, but not reliably, the reading order. MuPDF's own `sort=True`
sorts by (y, x), which interleaves two columns line by line.

**What we do on top.** Measure the text area; any block wider than 62 % of it is a
"span" block (title, section banner) that splits the page into horizontal bands. Among
the narrower blocks, merge their x-intervals; every gap wider than 12 pt is a gutter, so
the number of columns is discovered, not assumed. Emit band by band, column by column.
A second pass merges paragraph fragments MuPDF split (same column, tiny gap, same font
size), including a centred last caption line. Roles (heading, caption, header/footer,
body) come from font size, boldness and regexes.

**Why PyMuPDF.** Exact glyph decoding (it resolved the `≥` and `•` glyphs pdfminer and
docling-parse rendered as `‡` and `(cid:127)`), bboxes for everything, image placement
rectangles, rendering for OCR, one C library, fast (≈0.7 s/page including label OCR).

**Where it breaks on this corpus.** The UAV patent's page 1 scores 0.857 on reading-order
pairs: the abstract and the figure caption sit in the left column below a left-only title
block, and the right-column header line is emitted between them. Docling's learned
layout scores 0.929 on the same page. Fix with more time: treat per-column header lines
as band separators only within their column.

## Tables — pdfplumber for the grid, PyMuPDF for the text (`tables.py`)

**What pdfplumber does.** It reads vector graphics (`lines`, `rects`) and characters
via pdfminer.six. `find_tables` with the *lines* strategy takes every ruling line and
rect edge as a candidate edge, snaps edges within 3 pt and joins collinear segments,
intersects horizontals with verticals to get cell corners, forms the smallest rectangles
bounded by intersections as cells, and groups cells sharing edges into tables. The
*text* strategy synthesises edges from aligned word boundaries instead.

**What we do differently.** The grid comes from pdfplumber; each cell's text is the set
of PyMuPDF *words* whose centre lies in the cell, ordered by visual line. Two reasons:
pdfminer's glyph decoding (above), and pdfplumber assigns *characters* to cells, so when
text overflows a cell the two cells' characters interleave. The text strategy runs only
when no ruled table exists and only if no column edge cuts through a word; on prose it
otherwise "finds" a 19×8 table on a patent page. Rows whose first cell is empty are
wrapped continuations and are merged upward.

**Where it breaks.** Falcon-VT1 "Processor" row: the PDF itself draws "MHz" and "with" at
the same coordinates, so the Value cell physically overflows into Notes. We return
"Dual-core Cortex-M7 @ 480" / "MHz with M4 co-processor" (2 of 30 cells wrong, 0.983
overall); every parser we tried, Docling included, returns the same split. A fix would
need semantics (units belong with their number), which we chose not to encode.
Scanned tables are not detected at all by this backend (no ruling lines in a bitmap); on
OmniDocBench's 19 scanned tables native scores TEDS 0.0 against Docling's 0.53, which is
the main reason `hybrid` exists.

## Figures — PyMuPDF image rectangles + caption linking (`figures.py`)

**What the library does.** `page.get_image_info()` walks the display list and reports
every raster image draw with the rectangle it was painted into, so the box is correct
even when the image is scaled. `page.cluster_drawings()` groups nearby vector paths into
rectangles for line-art figures.

**What we do.** Keep images ≥ 60 pt on a side; drop drawing clusters that overlap a
detected table (table rulings cluster too). Captions are text blocks matching
`FIG. n —` / `Figure n —`, nearest with horizontal overlap, preferring below; a bare
"FIG. 2" title directly above the drawing is absorbed into the figure; a caption at the
top of the next page is linked to an uncaptioned figure at the bottom of the previous
one (the BMS enclosure figure). Labels are read from the crop with Tesseract in
sparse-text mode and reconciled against the numerals the specification defines.

**Where it breaks.** Reference numerals: Tesseract reads "100/110/120" as "00/10/20"
because a leader line touches the "1" (label recall 0.39 vs 0.96 for the VLM
description), and the gripper's "160" is outside the embedded image entirely. We also
initially clipped figure boxes to their text column and lost labels that sit past the
column edge; the clip was removed. The numeral index is therefore built from text, and
OCR only confirms presence.

## OCR — Tesseract 5 (`ocr.py`)

**When it runs.** Per page: fewer than 30 text characters plus a page-covering image
(no text layer), or a text layer whose tokens fail a plausibility score (symbol soup,
`(cid:)`, U+FFFD, per-character splitting). Born-digital pages never touch it.

**What Tesseract does.** Otsu binarisation and connected components → page layout
analysis (`--psm 3`: tab-stop detection finds column gutters, blobs are grouped into
lines and blocks) → LSTM line recogniser with the English language model, returning
words with confidences and a block/paragraph/line hierarchy. We render at 300 dpi, turn
Tesseract paragraphs into blocks with point bboxes and run the same column/band ordering
as born-digital pages, so both paths produce identical structures.

**Where it breaks.** On the scanned patent it reads "gate 412" for "gate 112" in claim 2
and "sAcooperate" where the figure label "64" touches the column edge; page CER is still
0.13 %. Docling on the same page emits several header-area blocks twice ("PROGRAMMED
ARTICLE TRANSFER PROGRAMMED ARTICLE TRANSFER", the abstract, ...): CER 21.8 % with its
Tesseract driver and 22.7 % with its default EasyOCR engine, so the duplication comes from
overlapping layout regions on this page, not from the OCR driver. That measurement is why
`hybrid` keeps native's OCR text on scanned pages and takes only Docling's table and figure
boxes from them.

## Docling (`ae/extract/docling_backend.py`)

docling-parse (C++ on qpdf) extracts text cells and renders the page; an RT-DETR style
object detector ("heron", trained on DocLayNet) predicts labelled boxes (text, section
header, caption, table, picture, list item, page header/footer, formula, code); cells are
assigned to boxes by overlap; OCR runs on bitmap regions without cells; TableFormer (an
encoder–decoder transformer) turns each table crop into an OTSL token sequence plus cell
boxes, matched back to text cells; a rule-based reading-order predictor sorts boxes using
left-to-right / top-to-bottom "sees" relations. On this corpus it links the title OCR'd
*inside* a picture as its caption (3 of 9 captions right), strips claim numbers into a
separate marker field (restored in our mapping), and shares pdfminer's `≥` problem. It
wins where there is no PDF structure: OmniDocBench text similarity 0.837 vs 0.749,
reading order 0.979 vs 0.948, 36 of 50 figures vs 0.

## DOCX — python-docx + LibreOffice render (`docx.py`, `pagemap.py`)

The OOXML body is an ordered list of paragraphs and tables; cells are explicit, images
are the original bytes, captions are the next paragraph. Nothing is detected. Pages do
not exist until a layout engine renders the file, so LibreOffice renders it once (cached)
and each block is located in the render to learn its page; a table that breaks across
pages becomes one block per page with the header repeated. Known difference: LibreOffice
puts the RJA-40 spec table's first four rows on page 1, the dev set cites page 2 for
them (Word keeps the table with its heading); we cite every page a split table spans.
