"""Tests for the hybrid retriever: RRF + metadata filtering + gating (offline)."""
from __future__ import annotations

from app.rag.reranker import NoopReranker
from app.rag.retriever import HybridRetriever, RetrievalResult


# ---- RRF fusion ----------------------------------------------------------

def test_rrf_fuses_two_rankings(make_settings):
    r = HybridRetriever(None, None, None, NoopReranker(), {}, make_settings(rrf_k=60))
    fused = r._rrf(["a", "b", "c"], ["b", "d", "a"])
    assert set(fused[:2]) == {"a", "b"}  # appear in both -> top
    assert fused == ["b", "a", "d", "c"]


def test_rrf_single_ranking_preserves_order(make_settings):
    r = HybridRetriever(None, None, None, NoopReranker(), {}, make_settings(rrf_k=60))
    assert r._rrf(["x", "y", "z"], []) == ["x", "y", "z"]


# ---- metadata filter detection ------------------------------------------

def test_detect_filters_company_and_year(build_retriever):
    r = build_retriever()
    tickers, years = r._detect_filters("What are NVIDIA's risk factors in 2024?")
    assert tickers == {"NVDA"}
    assert years == {"2024"}


def test_detect_filters_multi_company(build_retriever):
    r = build_retriever()
    tickers, _ = r._detect_filters("Compare revenue of NVIDIA and Apple")
    assert tickers == {"NVDA", "AAPL"}


def test_detect_filters_unknown_year_ignored(build_retriever):
    # 1999 is a valid-looking year but not in the corpus -> not used as a filter
    r = build_retriever()
    _, years = r._detect_filters("Apple revenue in 1999")
    assert years == set()


# ---- end-to-end retrieve -------------------------------------------------

def test_retrieve_filters_to_mentioned_company(build_retriever):
    r = build_retriever(confidence_threshold=0.05)
    result = r.retrieve("NVIDIA risk factors competition GPU")
    assert isinstance(result, RetrievalResult)
    assert result.hits
    assert all(h.ticker == "NVDA" for h in result.hits)  # AAPL chunks filtered out
    assert result.filters["tickers"] == ["NVDA"]


def test_retrieve_multi_company_keeps_both(build_retriever):
    r = build_retriever(confidence_threshold=0.05)
    result = r.retrieve("Compare revenue of NVIDIA and Apple")
    tickers = {h.ticker for h in result.hits}
    assert {"NVDA", "AAPL"} <= tickers  # both companies retrieved for synthesis


def test_retrieve_gates_when_below_threshold(build_retriever):
    r = build_retriever(confidence_threshold=0.99)
    result = r.retrieve("NVIDIA risk factors")
    assert result.top_cosine < 0.99
    assert result.gated is True


def test_retrieve_not_gated_when_above_threshold(build_retriever):
    r = build_retriever(confidence_threshold=0.05)
    result = r.retrieve("NVIDIA risk factors competition GPU markets")
    assert result.gated is False
