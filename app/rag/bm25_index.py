"""Sparse keyword index (BM25).

Catches exact-term matches that dense embeddings miss — tickers, GAAP line items
("deferred revenue"), product names, dollar figures. Fused with dense results via
RRF in the retriever. Persisted as plain JSON (corpus + ids) and the BM25 index
is rebuilt on load — fast for a filings-scale corpus.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self._ids: list[str] = []
        self._corpus: list[str] = []
        self._bm25: BM25Okapi | None = None

    @property
    def _file(self) -> Path:
        return self.index_path / "bm25.json"

    def build(self, ids: list[str], texts: list[str]) -> None:
        self._ids = list(ids)
        self._corpus = list(texts)
        self._bm25 = BM25Okapi([tokenize(t) for t in self._corpus])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def persist(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps({"ids": self._ids, "corpus": self._corpus}), encoding="utf-8"
        )

    def load(self) -> bool:
        if not self._file.exists():
            return False
        data = json.loads(self._file.read_text(encoding="utf-8"))
        self.build(data["ids"], data["corpus"])
        return True
