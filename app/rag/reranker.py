"""Reranker — optional second-stage precision booster.

Default is ``NoopReranker`` (keep fusion order) so the project runs with no heavy
deps. Flip ``RERANKER_ENABLED=true`` to load a cross-encoder
(``sentence-transformers``) that re-scores each (query, chunk) pair directly —
much sharper than bi-encoder similarity, at the cost of a torch install.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.vector_store import Hit


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


class NoopReranker(Reranker):
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        return hits[:top_k]


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # lazy, optional dep

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        pairs = [(query, h.text) for h in hits]
        scores = self.model.predict(pairs)
        for h, s in zip(hits, scores):
            h.metadata["rerank_score"] = float(s)
        ranked = sorted(hits, key=lambda h: h.metadata["rerank_score"], reverse=True)
        return ranked[:top_k]


def get_reranker(settings) -> Reranker:
    if settings.reranker_enabled:
        try:
            return CrossEncoderReranker(settings.reranker_model)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"⚠  reranker disabled ({exc}); falling back to no-op.")
    return NoopReranker()
