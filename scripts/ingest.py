"""Ingestion entrypoint: corpus -> section-aware chunks -> indexes.

Run from the project root:

    python scripts/ingest.py

Steps:
  1. Load every filing in data/corpus/ (HTML/PDF/txt) with its metadata sidecar.
  2. Section-aware chunking (10-K Item structure) carrying company/ticker/FY/section.
  3. Embed with Gemini, build the dense vector store + BM25 sparse index.
  4. Persist everything (vectors, payloads, BM25, canonical chunks.json).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.bm25_index import BM25Index  # noqa: E402
from app.rag.chunking import chunk_corpus  # noqa: E402
from app.rag.embeddings import get_embedder  # noqa: E402
from app.rag.loaders import load_corpus  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402
from config import settings  # noqa: E402


def main() -> None:
    print(f"📂 Corpus: {settings.corpus_path}")
    docs = load_corpus(settings.corpus_path)
    if not docs:
        print("❌ No documents found. Run `python scripts/fetch_edgar.py` first.")
        return
    print(f"📄 Loaded {len(docs)} filings.")

    chunks = chunk_corpus(docs, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        print("❌ No chunks produced.")
        return

    # report chunk distribution per company and per section
    by_company = Counter(f"{c.ticker} FY{c.fiscal_year}" for c in chunks)
    print(f"✂  {len(chunks)} chunks:")
    for name, n in sorted(by_company.items()):
        print(f"     {name:<16} {n:>4} chunks")
    print("   sections seen:")
    for sec, n in Counter(c.section for c in chunks).most_common():
        print(f"     {n:>4}  {sec}")

    print("🔢 Embedding chunks with Gemini…")
    embedder = get_embedder(settings)
    vectors = embedder.embed_documents([c.text for c in chunks])

    ids = [c.chunk_id for c in chunks]
    payloads = [
        {
            "text": c.text,
            "ticker": c.ticker,
            "company": c.company,
            "fiscal_year": c.fiscal_year,
            "section": c.section,
            "source_url": c.source_url,
            "metadata": c.metadata,
        }
        for c in chunks
    ]

    vs = get_vector_store(settings)
    vs.reset()
    vs.add(ids, vectors, payloads)
    vs.persist()
    print(f"🧮 Dense vector store persisted ({len(ids)} vectors).")

    bm25 = BM25Index(settings.index_path)
    bm25.build(ids, [c.text for c in chunks])
    bm25.persist()
    print("🔤 BM25 sparse index persisted.")

    settings.index_path.mkdir(parents=True, exist_ok=True)
    (settings.index_path / "chunks.json").write_text(
        json.dumps([{"chunk_id": c.chunk_id, **p} for c, p in zip(chunks, payloads)], ensure_ascii=False),
        encoding="utf-8",
    )
    print('✅ Ingestion complete. Try:  python scripts/ask.py "What are NVIDIA\'s main risk factors?"')


if __name__ == "__main__":
    main()
