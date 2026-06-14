"""Tests for multi-format document loaders."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.loaders import clean_text, load_corpus, load_document


def test_clean_text_collapses_whitespace_keeps_paragraphs():
    raw = "Line   one\t\twith   spaces\n\n\n\nLine two"
    out = clean_text(raw)
    assert "Line one with spaces" in out
    assert "\n\n\n" not in out  # 3+ blank lines collapsed
    assert out.count("\n\n") == 1


def test_load_html_strips_tags_and_scripts(tmp_path: Path):
    f = tmp_path / "doc.html"
    f.write_text(
        "<html><head><style>.x{}</style></head><body>"
        "<script>alert(1)</script><h1>Item 1. Business</h1>"
        "<p>We sell chips.</p></body></html>",
        encoding="utf-8",
    )
    text = load_document(f).text
    assert "Item 1. Business" in text
    assert "We sell chips." in text
    assert "alert(1)" not in text  # script dropped
    assert ".x{}" not in text  # style dropped


def test_load_txt_with_sidecar_metadata(tmp_path: Path):
    (tmp_path / "NVDA_2024_10K.txt").write_text("Some filing text.", encoding="utf-8")
    (tmp_path / "NVDA_2024_10K.json").write_text(
        json.dumps({"ticker": "NVDA", "company": "NVIDIA Corporation", "fiscal_year": "2024"}),
        encoding="utf-8",
    )
    doc = load_document(tmp_path / "NVDA_2024_10K.txt")
    assert doc.text == "Some filing text."
    assert doc.metadata["ticker"] == "NVDA"
    assert doc.metadata["fiscal_year"] == "2024"


def test_unsupported_type_raises(tmp_path: Path):
    f = tmp_path / "data.xyz"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(f)


def test_load_corpus_skips_sidecars_and_loads_supported(tmp_path: Path):
    (tmp_path / "A.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "A.json").write_text("{}", encoding="utf-8")  # sidecar, not a doc
    (tmp_path / "B.html").write_text("<p>beta</p>", encoding="utf-8")
    docs = load_corpus(tmp_path)
    assert len(docs) == 2  # only .txt and .html, never the .json
