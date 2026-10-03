"""Text-layer quality scoring and the per-page OCR trigger."""

import pymupdf

from ae.extract.native.ocr import MIN_TEXT_QUALITY, needs_ocr, text_quality


def test_clean_prose_with_numerals_and_units_scores_high():
    text = "The control unit 160 drives the electromagnet 120a via a relay at 3.3 V and 10 Hz."
    assert text_quality(text) >= 0.85


def test_symbol_soup_scores_zero():
    assert text_quality("§¶ ±± ¤¤ ‡† ∂∑ ®© ¥€ ~~~ ^^ ¬¬") == 0.0


def test_vowel_less_fragments_score_zero():
    assert text_quality("xkcd qwrt plmn zzzz brrr") == 0.0


def test_text_split_per_character_scores_below_threshold():
    assert text_quality("T h e c o n t r o l u n i t d r i v e s") < MIN_TEXT_QUALITY


def test_glued_cid_placeholders_never_count():
    assert text_quality("(cid:12)(cid:34) (cid:56)(cid:78) (cid:90)(cid:11)") == 0.0


def test_standalone_cid_placeholders_never_count():
    assert text_quality("(cid:56) (cid:78) (cid:90) (cid:12)") < MIN_TEXT_QUALITY


def test_replacement_characters_never_count():
    assert text_quality("�� word") == 0.5


def test_empty_text_scores_zero():
    assert text_quality("") == 0.0
    assert text_quality("  ... ;; ") == 0.0


def _page_with_text(tmp_path, text: str) -> pymupdf.Page:
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    page.insert_textbox(pymupdf.Rect(20, 20, 280, 380), text, fontsize=10)
    path = tmp_path / "text.pdf"
    doc.save(path)
    return pymupdf.open(path)[0]


def _page_with_full_page_image(tmp_path) -> pymupdf.Page:
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 30, 40), 0)
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(pymupdf.Rect(0, 0, 300, 400), pixmap=pix)
    path = tmp_path / "scan.pdf"
    doc.save(path)
    return pymupdf.open(path)[0]


def test_born_digital_page_does_not_need_ocr(tmp_path):
    page = _page_with_text(tmp_path, "The control unit 160 drives the electromagnet 120a through a relay. " * 4)
    assert needs_ocr(page) is None


def test_image_only_page_needs_ocr_for_missing_text_layer(tmp_path):
    assert needs_ocr(_page_with_full_page_image(tmp_path)) == "no_text_layer"


def test_garbled_text_layer_needs_ocr_for_low_quality(tmp_path):
    page = _page_with_text(tmp_path, "xq zr §§ ¤¤ kj ql " * 5)
    assert needs_ocr(page) == "low_text_quality"


def test_blank_page_is_reported_as_missing_text_layer():
    page = pymupdf.open().new_page(width=300, height=400)
    assert needs_ocr(page) == "no_text_layer"
