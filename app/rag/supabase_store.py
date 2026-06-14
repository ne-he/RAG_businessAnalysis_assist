"""Supabase pgvector backend (production path).

Enabled with ``VECTOR_STORE=supabase``. Lazy-imports ``supabase`` so the default
local path needs no extra deps. Provision the table + RPC with the SQL in the
README (``match_filing_chunks`` returns cosine similarity = ``1 - distance``).

Note: metadata filtering (ticker / fiscal_year) is applied client-side after the
RPC for parity with the local store; for large corpora push the filter into the
SQL function instead.
"""
from __future__ import annotations

from app.rag.vector_store import Hit, VectorStore, _payload_to_hit


class SupabaseVectorStore(VectorStore):
    def __init__(self, settings):
        from supabase import create_client  # lazy, optional dep

        if not (settings.supabase_url and settings.supabase_key):
            raise RuntimeError("SUPABASE_URL / SUPABASE_KEY missing in .env.")
        self.client = create_client(settings.supabase_url, settings.supabase_key)
        self.table = settings.supabase_table

    def reset(self) -> None:
        self.client.table(self.table).delete().neq("chunk_id", "").execute()

    def add(self, ids, vectors, payloads) -> None:
        rows = [
            {
                "chunk_id": cid,
                "content": p["text"],
                "ticker": p.get("ticker", ""),
                "company": p.get("company", ""),
                "fiscal_year": str(p.get("fiscal_year", "")),
                "section": p.get("section", ""),
                "source_url": p.get("source_url", ""),
                "metadata": p.get("metadata", {}),
                "embedding": vec,
            }
            for cid, vec, p in zip(ids, vectors, payloads)
        ]
        for i in range(0, len(rows), 100):
            self.client.table(self.table).upsert(rows[i : i + 100]).execute()

    def search(self, query_vector, k: int, allowed_ids=None) -> list[Hit]:
        res = self.client.rpc(
            "match_filing_chunks",
            {"query_embedding": query_vector, "match_count": max(k * 4, k)},
        ).execute()
        hits: list[Hit] = []
        for row in res.data or []:
            cid = row["chunk_id"]
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            payload = {
                "chunk_id": cid,
                "text": row["content"],
                "ticker": row.get("ticker", ""),
                "company": row.get("company", ""),
                "fiscal_year": row.get("fiscal_year", ""),
                "section": row.get("section", ""),
                "source_url": row.get("source_url", ""),
                "metadata": row.get("metadata", {}),
            }
            hits.append(_payload_to_hit(payload, float(row.get("similarity", 0.0))))
            if len(hits) >= k:
                break
        return hits

    def persist(self) -> None:
        pass  # remote store — nothing to persist locally

    def load(self) -> bool:
        return True  # assume the table is already provisioned
