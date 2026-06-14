"""Vector store — pluggable backend.

``LocalVectorStore`` (default) keeps a normalized float32 matrix on disk and does
exact cosine search with NumPy: zero infrastructure, instant to run. It supports
an optional ``allowed_ids`` filter so the retriever can restrict search to a
single company / fiscal year before ranking. The ``SupabaseVectorStore`` stub
shows the production path (pgvector) and is enabled with ``VECTOR_STORE=supabase``.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Hit:
    chunk_id: str
    text: str
    ticker: str
    company: str
    fiscal_year: str
    section: str
    source_url: str
    score: float  # cosine similarity in [-1, 1]
    metadata: dict = field(default_factory=dict)

    def citation(self) -> str:
        """Rich citation, e.g. ``[NVDA FY2024 · Item 1A. Risk Factors]``."""
        fy = f"FY{self.fiscal_year}" if self.fiscal_year else ""
        head = " ".join(p for p in (self.ticker, fy) if p)
        return f"[{head} · {self.section}]" if self.section else f"[{head}]"


def _payload_to_hit(p: dict, score: float) -> Hit:
    return Hit(
        chunk_id=p["chunk_id"],
        text=p["text"],
        ticker=p.get("ticker", ""),
        company=p.get("company", ""),
        fiscal_year=str(p.get("fiscal_year", "")),
        section=p.get("section", ""),
        source_url=p.get("source_url", ""),
        score=score,
        metadata=p.get("metadata", {}),
    )


class VectorStore(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def add(self, ids, vectors, payloads) -> None: ...

    @abstractmethod
    def search(self, query_vector: list[float], k: int, allowed_ids=None) -> list[Hit]: ...

    @abstractmethod
    def persist(self) -> None: ...

    @abstractmethod
    def load(self) -> bool:
        """Return True if a persisted index was loaded."""


class LocalVectorStore(VectorStore):
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self._vectors: np.ndarray | None = None
        self._payloads: list[dict] = []

    @property
    def _vec_file(self) -> Path:
        return self.index_path / "vectors.npy"

    @property
    def _meta_file(self) -> Path:
        return self.index_path / "payloads.json"

    def reset(self) -> None:
        self._vectors = None
        self._payloads = []

    def add(self, ids, vectors, payloads) -> None:
        mat = np.asarray(vectors, dtype=np.float32)
        self._vectors = mat if self._vectors is None else np.vstack([self._vectors, mat])
        for cid, p in zip(ids, payloads):
            self._payloads.append({"chunk_id": cid, **p})

    def search(self, query_vector: list[float], k: int, allowed_ids=None) -> list[Hit]:
        if self._vectors is None or len(self._payloads) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        sims = self._vectors @ q  # vectors are pre-normalized => cosine

        if allowed_ids is not None:
            allowed = set(allowed_ids)
            mask = np.array([p["chunk_id"] in allowed for p in self._payloads])
            if not mask.any():
                return []
            sims = np.where(mask, sims, -np.inf)
            n_valid = int(mask.sum())
        else:
            n_valid = len(self._payloads)

        k = min(k, n_valid)
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [_payload_to_hit(self._payloads[int(i)], float(sims[int(i)])) for i in top]

    def persist(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(self._vec_file, self._vectors)
        self._meta_file.write_text(json.dumps(self._payloads), encoding="utf-8")

    def load(self) -> bool:
        if not (self._vec_file.exists() and self._meta_file.exists()):
            return False
        self._vectors = np.load(self._vec_file)
        self._payloads = json.loads(self._meta_file.read_text(encoding="utf-8"))
        return True


def get_vector_store(settings) -> VectorStore:
    if settings.vector_store == "supabase":
        from app.rag.supabase_store import SupabaseVectorStore  # lazy

        return SupabaseVectorStore(settings)
    return LocalVectorStore(settings.index_path)
