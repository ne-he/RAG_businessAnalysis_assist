"""End-to-end RAG pipeline: wires retriever + generator and exposes a clean API.

``build_pipeline`` loads all persisted indexes (vectors, BM25, chunk store) and
constructs the runtime objects. ``answer`` is blocking; ``stream_answer`` yields
structured SSE-friendly events (sources first, then tokens, then a done marker).

Retrieval and generation fail independently, and only generation needs the network.
When Gemini is unavailable, usually a free-tier 429, retrieval has already done the
expensive and interesting part of the work: it found the right sections of the right
filings. Throwing that away and showing nothing is the worst possible outcome,
especially mid-demo, so the pipeline degrades to an extractive answer quoted straight
from the retrieved chunks, with the same citations. It is visibly worse than a written
answer and says so, which is the point: the system tells you it is degraded rather
than going quiet.

Streaming makes this ordering-sensitive. ``sources`` is emitted before any token, so a
failure after that point leaves a client showing citations and then nothing at all. An
explicit ``error`` event exists for exactly that window.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from app.rag.bm25_index import BM25Index
from app.rag.embeddings import get_embedder
from app.rag.generator import GenerationFailed, Generator
from app.rag.reranker import get_reranker
from app.rag.retriever import HybridRetriever, RetrievalResult
from app.rag.vector_store import Hit, get_vector_store

# Shown above an extractive answer so nobody mistakes quoted filing text for a
# model-written one.
FALLBACK_NOTICE = (
    "The language model is unavailable right now, so the passages below are quoted "
    "directly from the filings rather than summarised. Citations are unaffected."
)


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


def _mark_partial_lead(text: str) -> str:
    """Flag a passage that begins mid-sentence instead of silently trimming it.

    Chunks carry the previous chunk's last characters for context continuity, so a
    quoted passage very often starts mid-word. Cutting forward to the first clean
    sentence would read better, but it is a quiet edit of a filing, and telling
    apart a real fragment ("elines, and...") from prose that simply starts with a
    lowercase word ("iPhone revenue rose...") needs a dictionary, not a heuristic.
    A leading ellipsis is honest, is what a quotation would use anyway, and cannot
    delete evidence.
    """
    text = text.lstrip()
    if text and not (text[0].isupper() or text[0].isdigit()):
        return f"... {text}"
    return text


def _trim_to_sentence(text: str, max_chars: int) -> str:
    """Collapse whitespace and cut at a sentence boundary rather than mid-word."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    stop = max(cut.rfind(". "), cut.rfind("; "))
    if stop > max_chars // 2:
        return cut[: stop + 1]
    return cut.rstrip() + " ..."


def _extractive_answer(retrieval: RetrievalResult, limit: int = 3, chars: int = 600) -> str:
    """Answer from the retrieved chunks alone, with no model in the loop.

    Deliberately quotes rather than paraphrases: without a model there is nothing
    that can safely compress a financial statement, and a wrong summary of a filing
    is worse than a long one.
    """
    if retrieval.gated or not retrieval.hits:
        return (
            "I could not find support for that in the filings I have indexed. "
            "This system only answers from the 10-K filings it has ingested."
        )

    parts: list[str] = []
    seen: set[tuple] = set()
    for hit in retrieval.hits:
        key = (hit.ticker, hit.fiscal_year, hit.section)
        if key in seen:
            continue
        seen.add(key)
        passage = _trim_to_sentence(_mark_partial_lead(hit.text), chars)
        parts.append(f"{passage}\n{hit.citation()}")
        if len(parts) >= limit:
            break
    return f"{FALLBACK_NOTICE}\n\n" + "\n\n".join(parts)


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, history=None) -> dict:
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(query)
        notice: str | None = None
        try:
            text = self.generator.generate(query, retrieval, history)
        except GenerationFailed as exc:
            notice = str(exc)
            text = _extractive_answer(retrieval)
        return {
            "answer": text,
            "sources": _sources_from_hits(retrieval.hits),
            "gated": retrieval.gated,
            "top_cosine": round(retrieval.top_cosine, 4),
            "filters": retrieval.filters,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "degraded": notice is not None,
            "notice": notice,
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
        streamed = 0
        try:
            for token in self.generator.stream(query, retrieval, history):
                streamed += 1
                yield {"type": "token", "text": token}
        except GenerationFailed as exc:
            # `recovered` tells the client whether what follows is a usable answer or
            # a half-written one it should mark as truncated.
            yield {"type": "error", "message": str(exc), "recovered": streamed == 0}
            if streamed == 0:
                yield {"type": "token", "text": _extractive_answer(retrieval)}
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
