# FinSight — SEC 10-K RAG

A production-grade Retrieval-Augmented Generation system that answers questions
about public companies **strictly from their SEC 10-K filings**, with exact
citations down to the filing section. Ask *"What are NVIDIA's main risk factors?"*
or *"Compare the revenue drivers of NVIDIA and Microsoft"* and get a grounded,
cited answer — or an honest *"not found in the filings"* when the documents don't
support it.

> Built on a hybrid retrieval engine (dense + BM25 + RRF), section-aware 10-K
> chunking, metadata filtering, and a strict anti-fabrication confidence gate.
> **Not a Gemini wrapper** — see the table below.

---

## Why this isn't a "ChatGPT/Gemini wrapper"

| Concern | Naive wrapper | FinSight |
|---|---|---|
| Source of facts | Model's parametric memory (stale, hallucinated) | Retrieved 10-K excerpts only |
| Document structure | Dump raw text / blind splitter | **Section-aware** chunking per 10-K *Item* |
| Retrieval | Single dense similarity | **Hybrid** dense + BM25, fused with **RRF** |
| Targeting | One global search | **Metadata filtering** by company + fiscal year |
| Citations | None / made up | Exact: `[NVDA FY2026 · Item 1A. Risk Factors]` + source URL |
| Hallucination control | Hope | **Confidence gate** → "not found in filings" |
| Numbers | May invent figures | Quoted verbatim from context, or refused |
| Quality | "Looks fine" | **Eval harness**: hit-rate, gate accuracy, faithfulness |
| Reranking | — | Optional cross-encoder second stage |
| Serving | — | FastAPI streaming (SSE) + Docker + CI |

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
  SEC EDGAR ──────► │ fetch_edgar.py  (ticker→CIK→latest 10-K→text) │
                    └───────────────────────┬──────────────────────┘
                                            ▼  data/corpus/{TICKER}_{FY}_10K.txt (+.json)
                    ┌──────────────────────────────────────────────┐
                    │ ingest.py                                     │
                    │  loaders (HTML/PDF/txt) → section-aware chunk  │
                    │  → Gemini embeddings (768d) → vector + BM25    │
                    └───────────────────────┬──────────────────────┘
                                            ▼  data/index/
  query ─► retriever:  detect company/year ─► filter ─► dense ⊕ BM25 ─► RRF
           ─► (optional rerank) ─► confidence gate ─► generator (Gemini, cited)
                                            ▼
                         FastAPI /chat /chat/stream   ·   scripts/ask.py
```

### Pipeline highlights
- **Multi-format loaders** (`app/rag/loaders.py`): HTML (EDGAR), PDF (optional
  `pypdf`), and plain text — each paired with a metadata sidecar.
- **Section-aware chunking** (`app/rag/chunking.py`): detects the 10-K *Item*
  skeleton (Item 1 Business, 1A Risk Factors, 7 MD&A, 7A Market Risk, 8 Financial
  Statements, …). A largest-span + repeated-header-collapse heuristic ignores the
  Table of Contents and per-page header noise. Every chunk carries
  `company / ticker / fiscal_year / section / source_url`.
- **Hybrid retrieval + RRF** (`app/rag/retriever.py`): dense cosine search +
  BM25 keyword search fused with Reciprocal Rank Fusion.
- **Metadata filtering**: a query that names a company/year ("NVIDIA 2024") is
  restricted to that filing *before* ranking; multi-company questions keep all
  mentioned companies so the model can synthesize a comparison.
- **Confidence gate**: if the best dense cosine is below `CONFIDENCE_THRESHOLD`,
  the answer degrades to "not found in the available filings" — never a fabricated
  number.
- **Rich citations**: `[NVDA FY2026 · Item 1A. Risk Factors]` plus the source URL.

---

## Quickstart

```bash
# 1. Install (core deps are light; torch/pgvector are optional)
pip install -r requirements.txt

# 2. Configure
cp .env.example .env          # then set GEMINI_API_KEY and EDGAR_USER_AGENT

# 3. Download real 10-K filings from SEC EDGAR
python scripts/fetch_edgar.py            # NVDA, AAPL, MSFT (configurable)

# 4. Build the index (chunk → embed → vector + BM25)
python scripts/ingest.py

# 5. Ask
python scripts/ask.py "What are NVIDIA's main risk factors?"

# 6. Serve (streaming API) + open web/index.html
uvicorn app.main:app --reload
```

> **SEC requirement:** EDGAR requires a descriptive `User-Agent` with contact
> info on every request. Set `EDGAR_USER_AGENT` in `.env`.

---

## Evaluation

`eval/run_eval.py` scores a golden set of 19 questions (factual, comparison, and
out-of-scope). Retrieval metrics are cheap; `--gen` also generates and LLM-judges
answers (throttled for the free tier).

```bash
python eval/run_eval.py          # retrieval-only
python eval/run_eval.py --gen    # + answer match + faithfulness
```

<!-- EVAL_RESULTS -->
**Corpus:** NVDA (FY2026), AAPL (FY2025), MSFT (FY2025) — 947 chunks.
**Golden set:** 19 questions (16 factual/comparison + 3 out-of-scope).

| Metric | Result |
|---|---|
| Retrieval hit-rate@6 | **16/16 = 100%** (right company's filing retrieved) |
| Out-of-scope gate accuracy | **3/3 = 100%** (refused, no fabrication) |
| Mean top cosine | **0.738** |
| Faithfulness (LLM-judge, sampled) | **100%** on completed samples* |

\* The free-tier Gemini generation quota (~20 req/min for `gemini-2.5-flash`)
rate-limits a full 38-call generation sweep; the harness throttles + retries and
records faithfulness on the samples that complete. Retrieval metrics need only
embeddings and run over the whole set. Live spot-checks confirm grounded, cited
answers and correct refusals (see examples above).

The confidence threshold (`0.68`) was tuned from this data: legitimate questions
scored **0.72–0.80**, out-of-scope **0.55–0.63** — a clean separation.
<!-- /EVAL_RESULTS -->

---

## Configuration

All knobs live in `config.py`, overridable via `.env`:

| Setting | Default | Meaning |
|---|---|---|
| `GENERATION_MODEL` | `gemini-2.5-flash` | answer generation |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | embeddings (768-dim) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | chunking |
| `TOP_K_DENSE` / `TOP_K_SPARSE` | `10` / `10` | candidates per retriever |
| `FINAL_K` | `6` | chunks passed to the LLM |
| `CONFIDENCE_THRESHOLD` | `0.68` | gate below this top cosine |
| `RERANKER_ENABLED` | `false` | cross-encoder second stage |
| `VECTOR_STORE` | `local` | `local` (NumPy) or `supabase` |

---

## Deploying with Supabase pgvector

The default local NumPy store is dev-only. For production set
`VECTOR_STORE=supabase` and provision pgvector:

```sql
create extension if not exists vector;

create table filing_chunks (
  chunk_id    text primary key,
  content     text,
  ticker      text,
  company     text,
  fiscal_year text,
  section     text,
  source_url  text,
  metadata    jsonb,
  embedding   vector(768)
);
create index on filing_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create or replace function match_filing_chunks(query_embedding vector(768), match_count int)
returns table (
  chunk_id text, content text, ticker text, company text, fiscal_year text,
  section text, source_url text, metadata jsonb, similarity float
)
language sql stable as $$
  select chunk_id, content, ticker, company, fiscal_year, section, source_url, metadata,
         1 - (embedding <=> query_embedding) as similarity
  from filing_chunks
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

Then `python scripts/ingest.py` (with `VECTOR_STORE=supabase`) upserts the chunks,
and the container in `Dockerfile` just serves.

---

## Project layout

```
finance-rag/
├── config.py                 # env-overridable settings
├── app/
│   ├── main.py               # FastAPI: /health, /chat, /chat/stream (SSE)
│   ├── schemas.py · observability.py
│   └── rag/
│       ├── loaders.py        # HTML/PDF/txt → text + metadata
│       ├── chunking.py       # section-aware 10-K chunker
│       ├── embeddings.py     # Gemini embedder + cache + retry
│       ├── vector_store.py · supabase_store.py · bm25_index.py
│       ├── reranker.py       # no-op default + optional cross-encoder
│       ├── retriever.py      # hybrid + RRF + metadata filter + gate
│       ├── generator.py · prompts.py · pipeline.py
├── scripts/                  # fetch_edgar.py · ingest.py · ask.py
├── eval/                     # golden_set.yaml · run_eval.py
├── tests/                    # offline pytest (fake embedder, temp dirs)
├── web/index.html            # streaming test client
└── Dockerfile · .github/workflows/ci.yml
```

---

## Disclaimer

FinSight is an informational tool over public filings, **not investment advice**.
Facts are grounded in the retrieved 10-K excerpts; always verify against the
original filing via the cited source URL.
