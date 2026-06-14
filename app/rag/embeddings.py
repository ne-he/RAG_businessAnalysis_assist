"""Embedding layer — pluggable, with a disk cache and retry/backoff.

``Embedder`` is the interface; ``GeminiEmbedder`` is the default using Gemini
``gemini-embedding-001`` at ``output_dimensionality=768`` via the modern
``google-genai`` SDK. Embeddings are L2-normalized so a dot product equals cosine
similarity downstream. A JSON disk cache keyed by ``sha1(model+task+dim+text)``
avoids re-paying for identical text across re-ingests and eval runs.
"""
from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


def _normalize(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder(Embedder):
    def __init__(self, api_key: str, model_id: str, dim: int, cache_path: Path | None = None):
        from google import genai  # lazy: only when actually embedding

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is empty. Set it in .env before ingesting/querying."
            )
        self._client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.dim = dim
        self._cache_path = cache_path
        self._cache: dict[str, list[float]] = {}
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}

    def _key(self, text: str, task: str) -> str:
        raw = f"{self.model_id}|{task}|{self.dim}|{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")

    def _embed(self, text: str, task: str) -> list[float]:
        key = self._key(text, task)
        if key in self._cache:
            return self._cache[key]
        from google.genai import types

        # Retry transient failures (network blips / DNS hiccups / 429 / 503) with
        # exponential backoff so a single flake doesn't kill a long ingest. Bail
        # immediately on permanent client errors (bad model, auth, bad request).
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                res = self._client.models.embed_content(
                    model=self.model_id,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task,
                        output_dimensionality=self.dim,
                    ),
                )
                vec = _normalize(list(res.embeddings[0].values))
                self._cache[key] = vec
                return vec
            except Exception as exc:  # noqa: BLE001 — broad on purpose
                code = getattr(exc, "code", None)
                if code in (400, 401, 403, 404):
                    raise  # permanent: retrying won't help
                last_exc = exc
                if attempt < 4:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s
        raise last_exc  # type: ignore[misc]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i, t in enumerate(texts):
            out.append(self._embed(t, "RETRIEVAL_DOCUMENT"))
            if (i + 1) % 25 == 0:
                self._save_cache()
                print(f"    embedded {i + 1}/{len(texts)} chunks")
        self._save_cache()
        return out

    def embed_query(self, text: str) -> list[float]:
        vec = self._embed(text, "RETRIEVAL_QUERY")
        self._save_cache()
        return vec


def get_embedder(settings) -> Embedder:
    cache = settings.index_path / "embedding_cache.json"
    return GeminiEmbedder(
        api_key=settings.gemini_api_key,
        model_id=settings.embedding_model,
        dim=settings.embedding_dim,
        cache_path=cache,
    )
