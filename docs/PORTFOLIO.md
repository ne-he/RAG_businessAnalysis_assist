# FinSight — Portfolio Report

> A production-grade RAG system that answers questions about public companies
> strictly from their SEC 10-K filings, with section-level citations and a strict
> anti-fabrication gate. Built solo with Python + Gemini.

This document is written for a portfolio / interview context: it explains **what
the system does, how it's built, the non-obvious engineering decisions, and the
measured results.**

---

## 1. One-line pitch

> "I built a financial-analysis chatbot that answers questions from real SEC 10-K
> filings — hybrid retrieval, company/year metadata filtering, and a confidence
> gate that makes it say *'not found in the filings'* instead of hallucinating a
> number. It's measured with an eval harness, not vibes."

---

## 2. The problem it solves

LLMs are confidently wrong about financial facts — they hallucinate revenue
numbers, mix up fiscal years, and cite nothing. For anything finance-related that
is unacceptable. FinSight grounds every company-specific claim in the actual
filing text and cites it down to the section (e.g. *Item 1A. Risk Factors*), and
refuses when the documents don't support an answer.

---

## 3. What it does (demo flow)

| Ask | Behaviour |
|---|---|
| "What are NVIDIA's main risk factors?" | Detects company → filters to NVDA → grounded answer, every claim tagged `[NVDA FY2026 · Item 1A. Risk Factors]` |
| "Compare revenue drivers of NVIDIA and Microsoft" | Keeps **both** companies in retrieval → synthesizes a grounded comparison |
| "What is Tesla's stock price right now?" | Out of scope → **gated** → "not found in the filings", no fabrication |

---

## 4. Architecture

```
SEC EDGAR ─► fetch_edgar.py ─► data/corpus/{TICKER}_{FY}_10K.txt (+ .json metadata)
                                   │
        loaders (HTML/PDF/txt) ─► section-aware 10-K chunking ─► Gemini embeddings (768d)
                                   │                                   │
                                   ▼                                   ▼
                            BM25 sparse index                  NumPy vector store
                                   └──────────────┬────────────────────┘
   query ─► detect company/year ─► metadata filter ─► dense ⊕ BM25 ─► RRF fusion
         ─► (optional cross-encoder rerank) ─► confidence gate ─► Gemini (cited answer)
                                   │
                          FastAPI /chat/stream (SSE) · CLI · web client
```

**Stack:** Python 3.14 · Gemini (`gemini-2.5-flash` generation, `gemini-embedding-001`
@ 768-dim) via the modern `google-genai` SDK · `rank-bm25` · NumPy · FastAPI ·
optional Supabase pgvector · optional cross-encoder reranker.

---

## 5. The six things that make it more than a "wrapper"

1. **Multi-format document loaders** — HTML (EDGAR), PDF, and text, each with a
   metadata sidecar so provenance travels with the content.
2. **Section-aware 10-K chunking** — detects the *Item* skeleton (Business, Risk
   Factors, MD&A, Market Risk, Financial Statements…). Every chunk carries
   `company / ticker / fiscal_year / section / source_url`.
3. **Hybrid retrieval + RRF** — dense semantic search *and* BM25 keyword search
   (for tickers, GAAP line items, exact figures) fused with Reciprocal Rank Fusion.
4. **Metadata filtering** — "NVIDIA 2024" restricts search to that filing before
   ranking; multi-company questions keep all mentioned companies for comparison.
5. **Confidence gate (anti-fabrication)** — below a tuned cosine threshold the
   system answers "not found in the filings" rather than risk a fabricated number.
6. **Evaluation harness** — a golden set scored for retrieval hit-rate, gate
   accuracy, answer fact-match, and faithfulness (LLM-as-judge).

---

## 6. Engineering decisions & war-stories (the interview gold)

**Section detection vs. the Table of Contents.** A 10-K lists every *Item* twice:
once in the Table of Contents and again as the real section. Worse, the financial
statements stamp "Item 8" on *every page*, so a naive boundary detector shattered
Microsoft's filing into tiny fragments — it captured only **15%** of the document.
Fix: collapse consecutive repeats of the same Item heading, then keep the
**largest** span per Item. Coverage jumped to **88–92%** across all three filings
(Microsoft: 53 → 378 chunks).

**Tuning the confidence gate from data, not guesswork.** I plotted the top-cosine
of every eval question. Legitimate questions clustered at **0.72–0.80**;
out-of-scope ones at **0.55–0.63**. There was a clean gap, so I set the threshold
at **0.68** — out-of-scope questions gate, real questions don't. Measured, not
guessed.

**The deprecated-SDK / dead-model trap.** Google's `google.generativeai` SDK is
deprecated and `text-embedding-004` returns 404 on current API versions. I
migrated to the modern `google-genai` SDK and `gemini-embedding-001`, pinning
`output_dimensionality=768` to keep the pgvector schema stable.

**Free-tier resilience.** Generation is rate-limited to ~5 req/min on the free
tier. The embedder retries with exponential backoff; the eval harness throttles
and is resilient per-item so a single transient network blip doesn't void a whole
run. Embeddings are disk-cached so re-ingests are nearly free.

**Local-first, production-ready.** Default vector store is a zero-infra NumPy
index (cosine) so anyone can clone and run. Flip `VECTOR_STORE=supabase` for
pgvector — the table + RPC SQL ships in the README.

**Overlap applied twice, and the obvious fix was the wrong one.** When generation is
unavailable the system answers by quoting the retrieved chunks. The first time I
read that output, every passage opened by repeating its own first 200 characters and
then continuing. The splitter has two paths: paragraph assembly, and a sliding
window for a single paragraph longer than the chunk size. The window steps by
`size - overlap`, and a later stitching pass prepended the previous chunk's tail to
*every* chunk, so windowed chunks received the overlap twice. Diffing the rebuilt
index against the old one: **532 of 947 chunks opened with a verbatim copy of their
own first 200 characters, 123,381 characters or 11.0% of the corpus** was duplicated
text. Invisible in citations, which is why it survived, but embedded, BM25-indexed
and spent as context on every query.

The obvious fix is to stop stepping back and let the stitching handle it. I did
that, re-ingested, and re-ran the eval: **retrieval hit-rate fell from 16/16 to
14/16, and both losses were comparison questions.** The stride was not redundant
with the stitching, it was doing real work: overlapping windows are what keep a fact
sitting near a chunk boundary whole inside the next window, and comparison questions
need coverage of two companies inside the same top-k. So the stride stays and the
stitching yields instead, skipping any chunk that already overlaps its predecessor.
Three tests pin the result: no chunk repeats its own opening, consecutive chunks
still overlap exactly once, and reassembling the chunks reproduces the source text.

The generalisable part is that the eval harness caught this, not review. A change
that is obviously correct locally can still cost two points of recall, and without a
number attached to the old behaviour there is no way to know.

All three configurations are measured end to end, not argued:

| Chunking | Chunks | Corpus chars | Hit-rate@6 | Mean top cosine |
|---|---|---|---|---|
| Original (overlap applied twice) | 947 | 1,121,666 | 16/16 | 0.738 |
| Obvious fix (stride removed) | 849 | not kept | **14/16** | not kept |
| Shipped (stride kept, stitching yields) | 947 | 998,285 | 16/16 | **0.741** |

The shipped row is the interesting one: identical chunk boundaries to the original,
123,381 fewer characters, and the same perfect hit-rate with a slightly *higher* mean
cosine. Removing text that a chunk had already said sharpened its embedding a little
rather than costing recall.

**A demo that degrades instead of dying.** The free tier allows only a few
generations per minute, so a 429 mid-demo is routine. The SSE contract emits
`sources` *before* the first token, and the browser client only handled a failed
`fetch`, not a stream that died after a successful one. The failure mode was
therefore the worst available: citations render, then nothing, forever, with no
error anywhere. Now transient failures retry once, and if generation is still
unreachable the answer falls back to passages quoted from the chunks retrieval
already found, labelled as degraded and still fully cited. The stream emits an
explicit `error` event carrying a reason that never includes the API key. The retry
window deliberately closes after the first token, because replaying a stream that
has already reached the screen would duplicate text. `/health` returns 503 instead
of a decorative `{"status": "ok"}` when the pipeline failed to build.

---

## 7. Measured results

Corpus: latest 10-K for **NVDA (FY2026), AAPL (FY2025), MSFT (FY2025)** —
**947 chunks**, 88–92% text coverage.

<!-- RESULTS_TABLE -->
Golden set: **19 questions** (16 factual/comparison + 3 out-of-scope).

| Metric | Result | What it proves |
|---|---|---|
| Retrieval hit-rate@6 | **100%** (16/16) | Hybrid retrieval + metadata filtering route to the right filing every time |
| Out-of-scope gate accuracy | **100%** (3/3) | The system refuses unanswerable questions — no hallucinated numbers |
| Mean top cosine | **0.741** | Healthy separation from the 0.68 gate |
| Faithfulness (LLM-judge) | **100%** on completed samples | Generated answers are supported by the retrieved context |

> Note: the free Gemini tier rate-limits generation (~20/min), so a full 38-call
> faithfulness sweep is throttled; retrieval metrics (embeddings only) run over the
> whole set. The threshold 0.68 was chosen from the measured cosine gap
> (legit 0.72–0.80 vs out-of-scope 0.55–0.63).
<!-- /RESULTS_TABLE -->

---

## 8. What I'd do next

- Push metadata filtering into the pgvector SQL function for scale.
- Add per-page numeric cross-checks (regex-validate quoted figures against the chunk).
- Expand the corpus to multiple fiscal years per company for trend questions.
- Enable the cross-encoder reranker behind a flag for a precision boost.

---

## 9. How to run

```bash
pip install -r requirements.txt
cp .env.example .env                 # set GEMINI_API_KEY + EDGAR_USER_AGENT
python scripts/fetch_edgar.py        # download real 10-K filings
python scripts/ingest.py             # chunk → embed → index
pytest -q                            # unit tests
```

Then, for every run after that:

```bash
python demo.py                       # preflight, serve API + UI, open a browser
```
