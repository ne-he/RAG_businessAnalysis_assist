"""Hybrid retriever: metadata-filtered dense + sparse, fused with RRF, gated.

Pipeline per query:
  1. Parse the query for company/ticker mentions and fiscal years; build an
     ``allowed`` chunk-id set so "NVIDIA 2024" only ranks NVDA FY2024 chunks.
     Multiple companies stay allowed (so "compare X vs Y" retrieves both).
  2. Dense cosine search (restricted to ``allowed``) + BM25 sparse search.
  3. Fuse both rankings with Reciprocal Rank Fusion (RRF).
  4. Rerank the fused pool (optional cross-encoder).
  5. Gate on the best dense cosine — below threshold => ``gated`` so the
     generator answers "not found in the filings" instead of fabricating a number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.bm25_index import BM25Index
from app.rag.embeddings import Embedder
from app.rag.reranker import Reranker
from app.rag.vector_store import Hit, VectorStore

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_STOPWORDS = {"inc", "corp", "corporation", "co", "company", "ltd", "the", "and", "&"}


@dataclass
class RetrievalResult:
    hits: list[Hit]
    top_cosine: float
    gated: bool  # True => retrieval was weak, answer "not found in filings"
    filters: dict = field(default_factory=dict)  # {tickers: [...], years: [...]}


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25: BM25Index,
        reranker: Reranker,
        chunk_lookup: dict[str, Hit],
        settings,
    ):
        self.embedder = embedder
        self.vs = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.lookup = chunk_lookup
        self.s = settings
        self._alias_to_ticker, self._known_years = self._build_index(chunk_lookup)

    @staticmethod
    def _build_index(lookup: dict[str, Hit]) -> tuple[dict[str, str], set[str]]:
        """Map company-name tokens + ticker -> ticker, and collect known years."""
        alias: dict[str, str] = {}
        years: set[str] = set()
        for h in lookup.values():
            if h.ticker:
                alias[h.ticker.lower()] = h.ticker
            for tok in re.findall(r"[a-z0-9]+", (h.company or "").lower()):
                if tok and tok not in _STOPWORDS and len(tok) > 1:
                    alias.setdefault(tok, h.ticker)
            if h.fiscal_year:
                years.add(str(h.fiscal_year))
        return alias, years

    def _detect_filters(self, query: str) -> tuple[set[str], set[str]]:
        q = query.lower()
        tokens = set(re.findall(r"[a-z0-9]+", q))
        tickers = {self._alias_to_ticker[t] for t in tokens if t in self._alias_to_ticker}
        # _YEAR_RE has a capturing group, so use finditer/group(0) for the full year.
        years = {m.group(0) for m in _YEAR_RE.finditer(query)} & self._known_years
        return tickers, years

    def _allowed_ids(self, tickers: set[str], years: set[str]):
        if not tickers and not years:
            return None
        allowed = {
            cid
            for cid, h in self.lookup.items()
            if (not tickers or h.ticker in tickers)
            and (not years or str(h.fiscal_year) in years)
        }
        return allowed or None  # over-constrained -> don't filter, let the gate decide

    def _rrf(self, dense_ids: list[str], sparse_ids: list[str]) -> list[str]:
        scores: dict[str, float] = {}
        for ranking in (dense_ids, sparse_ids):
            for rank, cid in enumerate(ranking, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.s.rrf_k + rank)
        return sorted(scores, key=lambda c: scores[c], reverse=True)

    def retrieve(self, query: str) -> RetrievalResult:
        tickers, years = self._detect_filters(query)
        allowed = self._allowed_ids(tickers, years)

        # 1. dense (metadata-filtered)
        qvec = self.embedder.embed_query(query)
        dense_hits = self.vs.search(qvec, self.s.top_k_dense, allowed_ids=allowed)
        dense_cosine = {h.chunk_id: h.score for h in dense_hits}
        dense_ids = [h.chunk_id for h in dense_hits]
        top_cosine = dense_hits[0].score if dense_hits else 0.0

        # 2. sparse, then apply the same metadata filter
        sparse = self.bm25.search(query, self.s.top_k_sparse * 2)
        sparse_ids = [cid for cid, _ in sparse]
        if allowed is not None:
            sparse_ids = [cid for cid in sparse_ids if cid in allowed]
        sparse_ids = sparse_ids[: self.s.top_k_sparse]

        # 3. fuse
        fused_ids = self._rrf(dense_ids, sparse_ids)
        pool = max(self.s.final_k * 2, self.s.final_k)
        candidates: list[Hit] = []
        for cid in fused_ids[:pool]:
            hit = self.lookup.get(cid)
            if hit is None:
                continue
            hit.score = dense_cosine.get(cid, 0.0)  # carry dense cosine (0 if sparse-only)
            candidates.append(hit)

        # 4. rerank -> final_k
        final_hits = self.reranker.rerank(query, candidates, self.s.final_k)

        # 5. gate on best dense cosine
        gated = top_cosine < self.s.confidence_threshold
        return RetrievalResult(
            hits=final_hits,
            top_cosine=top_cosine,
            gated=gated,
            filters={"tickers": sorted(tickers), "years": sorted(years)},
        )
