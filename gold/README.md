# Extraction gold set

Hand-annotated ground truth for the extraction-stage evaluation (the dev set only
labels evidence pages, not extraction quality).

| File | Covers | How it was made |
|---|---|---|
| `ocr/US2988237A_programmed_article_transfer_p1.txt` | full text of the scanned patent's page 1 in reading order (figure labels excluded, caption included) | transcribed from the page image at 110 dpi by the author, re-read once against the image |
| `tables.json` | all four spec tables, cell by cell | design-doc PDFs read visually; RJA-40 taken from the DOCX XML (exact). The Falcon "Processor" row follows the *visually intended* cells, although the PDF overprints two cells |
| `figures.json` | every figure: page, id, caption (or prefix), caption page when different, reference numerals visible in the drawing | read from the figure crops / page images |
| `reading_order.json` | ordered paragraph anchors for three patent pages (two-column, one scanned) | from the page images |

Known corpus quirks recorded here rather than "fixed": the scanned patent calls 76 both a
"rate control" and a "valve"; the dev set cites page 3 for the BMS IP67 rating and page 2
for the scanned patent's anticipator, where the facts render on pages 2 and 1 respectively.
