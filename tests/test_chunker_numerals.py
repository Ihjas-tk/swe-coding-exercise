"""Numeral index and chunker against the dev corpus (native-backend extraction must exist).

Corpus-dependent tests are marked `integration` and skip unless the native page cache
(and the LibreOffice render cache for DOCX) has been built by `make ingest BACKEND=native`.
"""

from pathlib import Path

import pytest

from ae.index.chunker import chunk_document, structured_chunks
from ae.index.numerals import NumeralIndex, reconcile_labels
from ae.schema import ParsedDocument

ROOT = Path(__file__).resolve().parents[1]
CORPUS = sorted((ROOT / "patents").glob("*.pdf")) + sorted((ROOT / "design_docs").glob("*"))
_DOCS: dict[str, ParsedDocument] = {}


def _extraction_cached() -> bool:
    """True when every corpus file can be loaded from cache (no fresh OCR / LibreOffice run)."""
    from ae.extract.cache import CACHE_ROOT, PageCache, file_hash
    from ae.extract.native.pdf import VERSION

    if not CORPUS:
        return False
    for p in CORPUS:
        if p.suffix.lower() == ".pdf" and not (PageCache("native", VERSION, p).dir / "p1.json").exists():
            return False
        if (
            p.suffix.lower() == ".docx"
            and not (CACHE_ROOT / "render" / f"{p.stem}_{file_hash(p)}" / f"{p.stem}.pdf").exists()
        ):
            return False
    return True


needs_cache = pytest.mark.skipif(not _extraction_cached(), reason="run `make ingest BACKEND=native` first")


def load(name: str) -> ParsedDocument:
    """Parse through the native backend's page cache (fast after `make ingest`, correct before it)."""
    if name not in _DOCS:
        from ae.extract.base import parse

        path = next(p for p in CORPUS if p.stem == name)
        _DOCS[name] = parse(path, backend="native")
    return _DOCS[name]


@pytest.fixture(scope="module")
def idx() -> NumeralIndex:
    i = NumeralIndex()
    for p in CORPUS:
        if p.suffix.lower() in (".pdf", ".docx"):
            i.add_document(load(p.stem))
    return i


@pytest.mark.integration
@needs_cache
def test_dev_set_figure_numerals(idx):
    assert idx.lookup("US8485576B2_robotic_gripper.pdf", "160").name == "control unit"  # q13
    assert idx.lookup("US2988237A_programmed_article_transfer.pdf", "112").name == "gate"  # q14
    assert idx.lookup("US20170320570A1_uav_vtol.pdf", "18").name == "left thrust-generating device"  # q15


@pytest.mark.integration
@needs_cache
def test_suffixed_and_single_occurrence_numerals(idx):
    assert idx.lookup("US8485576B2_robotic_gripper.pdf", "120a").name == "electromagnet"
    assert idx.lookup("US8485576B2_robotic_gripper.pdf", "125a").name == "wire"


@pytest.mark.integration
@needs_cache
def test_design_docs_have_no_numerals(idx):
    for d in (
        "EV-BMS-100_design_document.pdf",
        "Falcon-VT1_flight_controller_design_document.pdf",
        "RJA-40_actuator_design_document.docx",
    ):
        assert idx.defined(d) == set(), d


@pytest.mark.integration
@needs_cache
def test_figure_defs(idx):
    f = idx.figures[("US8485576B2_robotic_gripper.pdf", "FIG. 2")]
    assert f.page == 1 and "cross section" in f.title


def test_reconcile_repairs_dropped_leading_digit():
    rec = reconcile_labels(["00", "10", "20", "50", "FIG."], {"100", "110", "120", "50", "160"})
    assert rec["matched"] == ["50"] and rec["repaired"] == ["100", "110", "120"]
    # ambiguous repair (two candidates) is left unmatched
    rec = reconcile_labels(["20"], {"120", "220"})
    assert rec["unmatched"] == ["20"]


@pytest.mark.integration
@needs_cache
def test_table_row_chunks(idx):
    chunks = chunk_document(load("EV-BMS-100_design_document"), idx)
    rows = [c for c in chunks if c.kind == "table_row"]
    hit = [c for c in rows if "Max discharge current" in c.text]
    assert len(hit) == 1 and "320 A" in hit[0].text and hit[0].page == 2 and "Electrical" in hit[0].text
    assert "320a" in hit[0].identifiers
    assert any(c.kind == "table" and c.page == 2 for c in chunks)


@pytest.mark.integration
@needs_cache
def test_figure_chunk_and_cross_page_caption(idx):
    chunks = chunk_document(load("EV-BMS-100_design_document"), idx)
    figs = [c for c in chunks if c.kind == "figure"]
    assert any("four M12" in c.text and c.page == 2 for c in figs)


@pytest.mark.integration
@needs_cache
def test_claims_and_glossary(idx):
    chunks = chunk_document(load("US8485576B2_robotic_gripper"), idx)
    claims = [c for c in chunks if c.kind == "claim"]
    assert claims and claims[0].meta["claim"] == 1
    fig = next(c for c in chunks if c.kind == "figure")
    assert "160 = control unit" in fig.text and "ref160" in fig.identifiers
    assert all(c.prefix.startswith("ROBOTIC GRIPPER") for c in chunks)
    assert all(c.page in (1, 2) for c in chunks)


@pytest.mark.integration
@needs_cache
def test_text_chunk_sizes(idx):
    for name in ("US20170320570A1_uav_vtol", "US2988237A_programmed_article_transfer"):
        for c in chunk_document(load(name), idx):
            if c.kind == "text":
                assert len(c.text.split()) <= 450 / 1.33 + 5, (name, c.id)


def test_structured_chunks(tmp_path):
    from ae.extract.structured import load_structured

    db = tmp_path / "t.sqlite"
    load_structured([ROOT / "structured" / "test_log.csv", ROOT / "structured" / "bill_of_materials.xlsx"], db)
    rows = structured_chunks(db)
    assert len(rows) == 109 + 25
    bom = next(c for c in rows if "FVT-ESC-40A" in c.text)
    assert "fvt-esc-40a" in bom.identifiers and bom.page == 1
