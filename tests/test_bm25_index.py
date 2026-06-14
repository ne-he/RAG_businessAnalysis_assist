"""Tests for the BM25 sparse index."""
from __future__ import annotations

from pathlib import Path

from app.rag.bm25_index import BM25Index

IDS = ["d1", "d2", "d3"]
TEXTS = [
    "Apple reports revenue from iPhone and services segments",
    "NVIDIA data center revenue from GPU accelerators and CUDA",
    "Microsoft cloud revenue from Azure and Office subscriptions",
]


def test_search_surfaces_doc_with_query_term(tmp_path: Path):
    idx = BM25Index(tmp_path)
    idx.build(IDS, TEXTS)
    results = idx.search("Azure", k=3)
    assert results
    assert results[0][0] == "d3"  # the Microsoft/Azure doc
    assert results[0][1] > 0


def test_search_empty_index_returns_empty(tmp_path: Path):
    assert BM25Index(tmp_path).search("anything", k=5) == []


def test_persist_load_round_trip(tmp_path: Path):
    idx = BM25Index(tmp_path)
    idx.build(IDS, TEXTS)
    before = idx.search("Azure", k=3)
    idx.persist()

    reloaded = BM25Index(tmp_path)
    assert reloaded.load() is True
    after = reloaded.search("Azure", k=3)
    assert [c for c, _ in before] == [c for c, _ in after]


def test_load_false_when_missing(tmp_path: Path):
    assert BM25Index(tmp_path).load() is False
