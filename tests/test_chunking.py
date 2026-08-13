"""Tests for section-aware 10-K chunking."""
from __future__ import annotations

from app.rag.chunking import _split_long, chunk_document, split_into_sections
from app.rag.loaders import LoadedDoc


def _toc() -> str:
    return (
        "Table of Contents\n"
        "Item 1. Business 1\n"
        "Item 1A. Risk Factors 5\n"
        "Item 7. MD&A 20\n"
        "Item 8. Financial Statements 40\n"
        "Item 9. Other 60\n\n"
    )


def _body() -> str:
    return (
        "Item 1. Business\n\n" + "We design and sell graphics processors. " * 20 + "\n\n"
        "Item 1A. Risk Factors\n\n" + "Competition is intense and demand may fall. " * 20 + "\n\n"
        "Item 7. Management's Discussion and Analysis\n\n"
        + "Revenue increased due to data center growth. " * 20 + "\n\n"
        "Item 8. Financial Statements\n\n"
        # repeated page-header noise that must be collapsed:
        + "Item 8.\n" * 30
        + "Total revenue was reported as a specific figure here. " * 20 + "\n\n"
        "Item 9. Other Information\n\nshort tail"
    )


# ---- _split_long ---------------------------------------------------------

def test_split_long_short_text_single_chunk():
    assert _split_long("short", 900, 150) == ["short"]
    assert _split_long("   ", 900, 150) == []


def test_split_long_splits_and_keeps_overlap():
    overlap, size = 30, 120
    text = "\n\n".join(["a" * 100, "b" * 100, "c" * 100])
    chunks = _split_long(text, size, overlap)
    assert len(chunks) >= 2
    assert chunks[1].startswith(chunks[0][-overlap:])


def test_long_paragraph_is_not_overlapped_twice():
    """A chunk must never open with a copy of its own first `overlap` characters.

    The sliding-window branch steps by `size - overlap`, and the stitching pass below
    it also prepended the previous chunk's tail, so the two overlaps landed on top of
    each other and every windowed chunk began by repeating itself verbatim. Measured
    on the built index this hit 458 of 947 chunks and 8.2% of all corpus text, which
    is embedded, BM25-indexed and sent to the model as context.

    Distinct characters are used instead of runs of "a" because repeated filler
    cannot tell a real overlap from a duplicated one.
    """
    overlap, size = 20, 100
    para = "".join(chr(ord("a") + i % 26) for i in range(500))
    chunks = _split_long(para, size, overlap)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk[:overlap] != chunk[overlap + 1 : 2 * overlap + 1], (
            f"chunk repeats its own opening {overlap} chars: {chunk[: 2 * overlap]!r}"
        )


def test_long_paragraph_chunks_still_overlap_once():
    """The fix must not remove overlap, only stop applying it twice."""
    overlap, size = 20, 100
    para = "".join(chr(ord("a") + i % 26) for i in range(500))
    chunks = _split_long(para, size, overlap)

    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.startswith(prev[-overlap:]), "context continuity was lost"


def test_long_paragraph_covers_the_whole_text():
    """No characters may be dropped between chunks."""
    overlap, size = 20, 100
    para = "".join(chr(ord("a") + i % 26) for i in range(500))
    chunks = _split_long(para, size, overlap)

    rebuilt = chunks[0] + "".join(c[overlap:] for c in chunks[1:])
    assert rebuilt.replace("\n", "") == para


# ---- split_into_sections -------------------------------------------------

def test_detects_major_items():
    sections = dict(split_into_sections(_toc() + _body()))
    names = list(sections)
    assert any("Risk Factors" in n for n in names)
    assert any("MD&A" in n for n in names)
    assert any("Financial Statements" in n for n in names)
    assert len(sections) >= 3


def test_body_wins_over_toc():
    sections = dict(split_into_sections(_toc() + _body()))
    risk = next(v for k, v in sections.items() if "Risk Factors" in k)
    assert "Competition is intense" in risk  # the long body, not the 1-line TOC entry


def test_collapses_repeated_headers_keeps_section_body():
    # The 30 repeated "Item 8." page headers must not fragment the section:
    # the revenue text that follows them must remain inside Item 8.
    sections = dict(split_into_sections(_toc() + _body()))
    fin = next(v for k, v in sections.items() if "Financial Statements" in k)
    assert "Total revenue was reported" in fin


def test_fallback_when_no_items():
    text = "Just some prose with no item headings at all. " * 30
    assert split_into_sections(text) == [("Full Filing", text)]


def test_fallback_when_too_few_sections():
    text = "Item 1. Business\n\n" + "x" * 600  # only one credible section
    out = split_into_sections(text)
    assert out == [("Full Filing", text)]


# ---- chunk_document ------------------------------------------------------

def test_chunk_document_carries_metadata():
    doc = LoadedDoc(
        text=_toc() + _body(),
        metadata={"ticker": "nvda", "company": "NVIDIA Corporation", "fiscal_year": "2024",
                  "source_url": "https://sec.gov/x.htm", "filing_type": "10-K"},
    )
    chunks = chunk_document(doc, size=1200, overlap=200)
    assert chunks
    c = chunks[0]
    assert c.ticker == "NVDA"  # upper-cased
    assert c.company == "NVIDIA Corporation"
    assert c.fiscal_year == "2024"
    assert c.source_url == "https://sec.gov/x.htm"
    assert c.chunk_id.startswith("NVDA_FY2024::")
    assert all(ch.section for ch in chunks)
