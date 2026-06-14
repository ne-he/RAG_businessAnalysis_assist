"""End-to-end RAG pipeline: wires retriever + generator and exposes a clean API.

``build_pipeline`` loads all persisted indexes (vectors, BM25, chunk store) and
constructs the runtime objects. ``answer`` is blocking; ``stream_answer`` yields
structured SSE-friendly events (sources first, then tokens, then a done marker).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from app.rag.bm25_index import BM25Index
from app.rag.embeddings import get_embedder
from app.rag.generator import Generator
from app.rag.reranker import get_reranker
from app.rag.retriever import HybridRetriever, RetrievalResult
from app.rag.vector_store import Hit, get_vector_store


def _load_chunk_lookup(index_path: Path) -> dict[str, Hit]:
    f = index_path / "chunks.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run `python scripts/ingest.py` first.")
    lookup: dict[str, Hit] = {}
    for c in json.loads(f.read_text(encoding="utf-8")):
        lookup[c["chunk_id"]] = Hit(
            chunk_id=c["chunk_id"],
            text=c["text"],
            ticker=c.get("ticker", ""),
            company=c.get("company", ""),
            fiscal_year=str(c.get("fiscal_year", "")),
            section=c.get("section", ""),
            source_url=c.get("source_url", ""),
            score=0.0,
            metadata=c.get("metadata", {}),
        )
    return lookup


def _sources_from_hits(hits: list[Hit]) -> list[dict]:
    """One entry per (ticker, fiscal_year, section), keeping the best score."""
    seen: dict[tuple, dict] = {}
    for h in hits:
        key = (h.ticker, h.fiscal_year, h.section)
        if key not in seen or h.score > seen[key]["score"]:
            seen[key] = {
                "ticker": h.ticker,
                "fiscal_year": h.fiscal_year,
                "section": h.section,
                "source_url": h.source_url,
                "citation": h.citation(),
                "score": round(h.score, 4),
            }
    return list(seen.values())


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, history=None) -> dict:
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(query)
        text = self.generator.generate(query, retrieval, history)
        return {
            "answer": text,
            "sources": _sources_from_hits(retrieval.hits),
            "gated": retrieval.gated,
            "top_cosine": round(retrieval.top_cosine, 4),
            "filters": retrieval.filters,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def retrieve_only(self, query: str) -> RetrievalResult:
        return self.retriever.retrieve(query)

    def stream_answer(self, query: str, history=None) -> Iterator[dict]:
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(query)
        yield {
            "type": "sources",
            "sources": _sources_from_hits(retrieval.hits),
            "gated": retrieval.gated,
            "top_cosine": round(retrieval.top_cosine, 4),
            "filters": retrieval.filters,
        }
        for token in self.generator.stream(query, retrieval, history):
            yield {"type": "token", "text": token}
        yield {"type": "done", "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def build_pipeline(settings) -> RAGPipeline:
    vs = get_vector_store(settings)
    if not vs.load():
        raise FileNotFoundError(
            "Vector index not found. Run `python scripts/ingest.py` first."
        )
    bm25 = BM25Index(settings.index_path)
    bm25.load()
    lookup = _load_chunk_lookup(settings.index_path)
    retriever = HybridRetriever(
        embedder=get_embedder(settings),
        vector_store=vs,
        bm25=bm25,
        reranker=get_reranker(settings),
        chunk_lookup=lookup,
        settings=settings,
    )
    generator = Generator(
        api_key=settings.gemini_api_key,
        model_name=settings.generation_model,
    )
    return RAGPipeline(retriever, generator)
