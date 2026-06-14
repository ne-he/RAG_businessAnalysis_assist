"""Tests for the local NumPy vector store (incl. metadata-filter search)."""
from __future__ import annotations

import math
from pathlib import Path

from app.rag.vector_store import LocalVectorStore

IDS = ["NVDA::a", "AAPL::b", "MSFT::c"]
_inv = 1.0 / math.sqrt(2)
VECTORS = [
    [1.0, 0.0, 0.0],
    [_inv, _inv, 0.0],
    [0.0, 1.0, 0.0],
]
PAYLOADS = [
    {"text": "nvda chunk", "ticker": "NVDA", "company": "NVIDIA", "fiscal_year": "2024",
     "section": "Item 1A. Risk Factors", "source_url": "u1"},
    {"text": "aapl chunk", "ticker": "AAPL", "company": "Apple", "fiscal_year": "2024",
     "section": "Item 7. MD&A", "source_url": "u2"},
    {"text": "msft chunk", "ticker": "MSFT", "company": "Microsoft", "fiscal_year": "2023",
     "section": "Item 1. Business", "source_url": "u3"},
]


def _store(tmp_path: Path) -> LocalVectorStore:
    vs = LocalVectorStore(tmp_path)
    vs.reset()
    vs.add(IDS, VECTORS, PAYLOADS)
    return vs


def test_search_orders_by_cosine_and_maps_fields(tmp_path: Path):
    vs = _store(tmp_path)
    hits = vs.search([1.0, 0.0, 0.0], k=3)
    assert [h.chunk_id for h in hits] == ["NVDA::a", "AAPL::b", "MSFT::c"]
    assert hits[0].score == 1.0
    assert hits[0].ticker == "NVDA"
    assert hits[0].section == "Item 1A. Risk Factors"
    assert hits[0].citation() == "[NVDA FY2024 · Item 1A. Risk Factors]"


def test_allowed_ids_filter_restricts_results(tmp_path: Path):
    vs = _store(tmp_path)
    hits = vs.search([1.0, 0.0, 0.0], k=3, allowed_ids={"AAPL::b"})
    assert len(hits) == 1
    assert hits[0].chunk_id == "AAPL::b"


def test_allowed_ids_empty_returns_empty(tmp_path: Path):
    vs = _store(tmp_path)
    assert vs.search([1.0, 0.0, 0.0], k=3, allowed_ids=set()) == []


def test_persist_load_round_trip(tmp_path: Path):
    vs = _store(tmp_path)
    before = vs.search([_inv, _inv, 0.0], k=3)
    vs.persist()
    reloaded = LocalVectorStore(tmp_path)
    assert reloaded.load() is True
    after = reloaded.search([_inv, _inv, 0.0], k=3)
    assert [h.chunk_id for h in before] == [h.chunk_id for h in after]
    assert after[0].chunk_id == "NVDA::a" or after[0].chunk_id == "AAPL::b"
    assert reloaded.search([1.0, 0.0, 0.0], k=1)[0].company == "NVIDIA"


def test_load_false_when_missing(tmp_path: Path):
    assert LocalVectorStore(tmp_path).load() is False
