# How extraction works

Extraction turns each file into pages of blocks (text, table, figure), each carrying its
document, page number and position. For PDFs I built the extraction from three small
libraries, PyMuPDF, pdfplumber and Tesseract, and use Docling's layout model for one job only:
finding tables and figures on scanned pages. I compared this with using Docling for everything
before settling on it; the numbers are in the README under "How the stack was chosen".

## Text and reading order: PyMuPDF

**Under the hood.** PyMuPDF walks the PDF's content stream and groups glyphs into lines and
blocks by position and font. Blocks come back in the order the file was written, which is
usually but not always the reading order, and its built-in sort interleaves two columns. So I
find the columns myself: a block wider than about 60 % of the page is a title or banner that
splits the page into bands, and the gaps between the narrower blocks in a band give the column
gutters. Text is emitted band by band, column by column.

**Why.** Exact glyph decoding (pdfminer and Docling's parser both turned "≥" and "•" into
junk on the design documents), a position for everything, page rendering for OCR, and one
fast library instead of a pipeline.

**Where it fails here.** Page 1 of the UAV patent has a title block over the left column only,
and the right column's header line gets read between the abstract and the caption below it:
reading order on that page scores 0.86. The fix is to let a header that belongs to one column
act as a separator for that column only.

## Tables: pdfplumber for the grid, PyMuPDF for the text

**Under the hood.** pdfplumber reads the ruling lines drawn on the page, snaps nearby edges
together, intersects horizontals with verticals and turns the smallest rectangles into cells.
I fill each cell with the PyMuPDF words whose centre falls inside it rather than pdfplumber's
own cell text, which interleaves characters when a value overflows. A row with an empty first
cell is a wrapped continuation and is merged upward. The fallback that guesses a grid from
aligned text runs only when there are no ruling lines and never when a guessed edge would cut
through a word, because on prose it otherwise finds tables that aren't there.

**Why.** Ruled lines are the truth for these design-document tables, and pdfplumber exposes
them directly. It gets 115 of the 117 gold cells right.

**Where it fails here.** The Falcon-VT1 "Processor" row: the PDF prints "MHz" and "with" on
top of each other, so the value overflows into the notes column and every parser splits it the
same way. I left it rather than teach the parser that units belong with their number. The real
gap is scanned tables: a bitmap has no ruling lines, so this finds nothing there, which is why
Docling's layout model runs on scanned pages.

## Figures: PyMuPDF image boxes and caption matching

**Under the hood.** PyMuPDF reports every image drawn on the page with the rectangle it was
painted into, and clusters nearby vector paths into rectangles for line drawings. I keep
anything bigger than a thumbnail, drop clusters that overlap a table, and link each figure to
the nearest "FIG. n" or "Figure n" caption that overlaps it horizontally, preferring the one
below; a caption at the top of the next page is linked to an uncaptioned figure at the bottom
of the previous one. Numerals on the drawing are read with Tesseract and checked against the
numerals the specification defines.

**Why.** The image rectangles are exact, so crops and citations are exact, and caption
matching by position needs nothing learned.

**Where it fails here.** Tesseract reads "100", "110" and "120" on the gripper drawing as
"00", "10" and "20" because a leader line touches the "1", and label "160" lies outside the
embedded image altogether. So what each numeral means is indexed from the specification
text, where every numeral is defined, and OCR only confirms what is visible.

## OCR: Tesseract 5

**Under the hood.** A page with no usable text layer is rendered at 300 dpi; Tesseract
binarises it, finds the column gutters and text blocks, and runs its line recogniser on each
line. Its paragraphs become positioned blocks that go through the same column-and-band
ordering as digital pages, so both kinds of page produce the same structure. A page goes to
OCR when it has almost no text, or when its text fails a plausibility check (symbol soup,
undecodable glyphs, letters split one per word).

**Why.** On the scanned patent it reads the page almost perfectly: 0.13 % character error
rate against a hand transcription. Docling's text on the same page duplicated several header
blocks (22 % error rate), and the duplication was the same with a different OCR engine inside
Docling, so the fault is in its layout regions, not its OCR. That is why Tesseract's text is
kept on scanned pages and only Docling's table and figure boxes are taken from them.

**Where it fails here.** On dense two-column scans from the OmniDocBench benchmark it scores
0.74 on text similarity where Docling's text scored 0.84. Finding and fixing the duplication
would make Docling's text the better choice for scanned pages.

## Scanned pages: Docling's layout model

Docling runs an object detector over the page image that labels regions (text, table,
picture, caption) and a second model that reads a table region cell by cell. It runs only for
PDFs with at least one OCR'd page, and only those pages' tables and figures are taken: a
Tesseract text block inside a Docling table is replaced by the table, and a Docling figure
that duplicates one already found is dropped. On the 106 benchmark pages it finds 37 of 50
figures and about half of each table's structure, against nothing without it.

## DOCX, CSV and XLSX

A DOCX is a list of paragraphs and tables with explicit cells, images and captions, so nothing
is detected. Pages only exist once something renders the file, so LibreOffice renders it once
and each block is located in the render to learn its page. LibreOffice and Word don't always
break pages in the same place: the RJA-40 spec table starts on page 1 here while the dev set
cites page 2, so a table that breaks across pages is cited with every page it spans. CSV and
XLSX files become typed SQLite tables; the header is the row with the most text cells and
column types follow what the values look like.
