"""Lightweight observability — append one JSON line per query.

Captures the signals you actually want when debugging a finance RAG in
production: the query, the top cosine, whether it was gated, the metadata filters
that were applied, which filings were cited, and latency. Tail
``logs/queries.jsonl`` to spot weak-retrieval queries that need more corpus.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log_query(log_path: Path, *, query: str, result: dict) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "gated": result.get("gated"),
        "top_cosine": result.get("top_cosine"),
        "filters": result.get("filters"),
        "n_sources": len(result.get("sources", [])),
        "sources": [s.get("citation") for s in result.get("sources", [])],
        "latency_ms": result.get("latency_ms"),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
