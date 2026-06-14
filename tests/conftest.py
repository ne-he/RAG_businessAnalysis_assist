"""Shared pytest fixtures and offline test doubles.

Everything here runs with NO network and NO Gemini API key. ``FakeEmbedder`` is a
deterministic bag-of-words hashing embedder (L2-normalized) so a dot product
behaves like cosine similarity — identical text ~1.0, disjoint text ~0.0. That
lets us exercise the real retrieval / filtering / gating logic offline.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.bm25_index import BM25Index  # noqa: E402
from app.rag.embeddings import Embedder  # noqa: E402
from app.rag.reranker import NoopReranker  # noqa: E402
from app.rag.retriever import HybridRetriever  # noqa: E402
from app.rag.vector_store import Hit, LocalVectorStore  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FakeEmbedder(Embedder):
    """Deterministic, offline embedder: hashed bag-of-words, L2-normalized."""

    def __init__(self, dim: int = 96):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN_RE.findall(text.lower()):
            bucket = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16) % self.dim
            v[bucket] += 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def make_settings():
    """Factory for a minimal settings object the retriever understands."""

    def _make(**overrides):
        base = dict(
            top_k_dense=10,
            top_k_sparse=10,
            rrf_k=60,
            final_k=6,
            confidence_threshold=0.3,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    return _make


# A tiny synthetic 2-company corpus used by retriever/vector-store tests.
# (chunk_id, text, ticker, company, fiscal_year, section)
CORPUS = [
    ("NVDA::risk::0", "Risk factors include competition in GPU and AI accelerator markets",
     "NVDA", "NVIDIA Corporation", "2024", "Item 1A. Risk Factors"),
    ("NVDA::mdna::0", "Revenue grew driven by data center demand for accelerated computing",
     "NVDA", "NVIDIA Corporation", "2024", "Item 7. Management's Discussion and Analysis (MD&A)"),
    ("AAPL::risk::0", "Risk factors include supply chain concentration and product cycles",
     "AAPL", "Apple Inc.", "2024", "Item 1A. Risk Factors"),
    ("AAPL::mdna::0", "Revenue from iPhone services and wearables across regions",
     "AAPL", "Apple Inc.", "2024", "Item 7. Management's Discussion and Analysis (MD&A)"),
]


@pytest.fixture
def corpus():
    return list(CORPUS)


def make_hit(chunk_id, text, ticker, company, fiscal_year, section) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        text=text,
        ticker=ticker,
        company=company,
        fiscal_year=fiscal_year,
        section=section,
        source_url=f"https://example.com/{ticker}.htm",
        score=0.0,
    )


@pytest.fixture
def build_retriever(fake_embedder, make_settings, tmp_path):
    """Factory: assemble a HybridRetriever over the synthetic CORPUS, offline."""

    def _build(**settings_overrides):
        ids = [c[0] for c in CORPUS]
        texts = [c[1] for c in CORPUS]
        payloads = [
            {"text": t, "ticker": tk, "company": co, "fiscal_year": fy,
             "section": se, "source_url": f"https://example.com/{tk}.htm"}
            for (cid, t, tk, co, fy, se) in CORPUS
        ]
        vs = LocalVectorStore(tmp_path)
        vs.reset()
        vs.add(ids, fake_embedder.embed_documents(texts), payloads)
        bm25 = BM25Index(tmp_path)
        bm25.build(ids, texts)
        lookup = {c[0]: make_hit(*c) for c in CORPUS}
        return HybridRetriever(
            fake_embedder, vs, bm25, NoopReranker(), lookup,
            make_settings(**settings_overrides),
        )

    return _build
