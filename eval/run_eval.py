"""Evaluation harness for the finance-RAG.

Measures what actually matters for a financial RAG:
  * Retrieval hit-rate@k  — was the right company's filing retrieved?
  * Gate accuracy         — are out-of-scope questions correctly flagged weak?
  * Answer match (--gen)  — does the generated answer contain expected facts?
  * Faithfulness (--gen)  — LLM-judge: is the answer supported by the context?

Generation is throttled (free-tier Gemini = ~5 req/min) and retries on 429.

Usage:
    python eval/run_eval.py            # retrieval-only (fast, cheap)
    python eval/run_eval.py --gen      # also generate + judge answers (slow)
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.pipeline import build_pipeline  # noqa: E402
from config import settings  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden_set.yaml"
GEN_THROTTLE_S = 13  # stay under free-tier 5 req/min for generation


def _retrieved_tickers(result) -> set[str]:
    return {h.ticker for h in result.hits}


def _retry_delay(exc) -> float | None:
    """Parse the server-suggested retry delay from a 429 ('retry in 33.6s')."""
    m = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(m.group(1)) if m else None


def _gen_with_retry(fn, *args):
    """Call a Gemini-backed function, honoring the server's 429 retry delay."""
    for attempt in range(6):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", None)
            if code in (400, 401, 403, 404) or attempt == 5:
                raise
            # honor the server's suggested delay (429), else exponential backoff
            wait = _retry_delay(exc) or (5 * (attempt + 1))
            time.sleep(min(75, wait + 1))
    return None


def _judge_faithfulness(client, model_name: str, question: str, answer: str, context: str) -> int:
    prompt = (
        "Judge whether the ANSWER is supported ONLY by the CONTEXT (no fabricated "
        "facts/numbers). Reply ONLY '1' if faithful/grounded, or '0' if any fact is "
        "unsupported.\n\n"
        f"QUESTION: {question}\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nVerdict (0/1):"
    )
    try:
        out = _gen_with_retry(
            lambda: client.models.generate_content(model=model_name, contents=prompt).text.strip()
        )
        return 1 if out and out.startswith("1") else 0
    except Exception:
        return -1


def main() -> None:
    do_gen = "--gen" in sys.argv
    items = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    pipeline = build_pipeline(settings)

    judge = None
    if do_gen:
        from google import genai

        judge = genai.Client(api_key=settings.gemini_api_key)

    hits = gate_ok = ans_ok = ans_total = faithful = faithful_total = 0
    cosines: list[float] = []

    print(f"\n{'id':<22}{'type':<13}{'retr':<6}{'cos':<7}{'gate':<6}{'ans':<5}")
    print("-" * 64)
    for it in items:
        rr = pipeline.retrieve_only(it["question"])
        cosines.append(rr.top_cosine)

        exp = set(it.get("expected_sources") or [])
        retr = "-"
        if exp:
            ok = exp <= _retrieved_tickers(rr)  # all expected companies retrieved
            hits += ok
            retr = "✓" if ok else "✗"

        gate = "-"
        if it["type"] == "out-of-scope":
            gate_ok += rr.gated
            gate = "✓" if rr.gated else "✗"

        ans = "-"
        if do_gen:
            try:
                out = _gen_with_retry(pipeline.answer, it["question"])
                answer = (out["answer"] or "").lower()
                must = [m.lower() for m in (it.get("must_include") or [])]
                if must:
                    ans_total += 1
                    ok = all(m in answer for m in must)
                    ans_ok += ok
                    ans = "✓" if ok else "✗"
                ctx = "\n".join(h.text for h in rr.hits)
                verdict = _judge_faithfulness(
                    judge, settings.generation_model, it["question"], out["answer"], ctx
                )
                if verdict in (0, 1):
                    faithful_total += 1
                    faithful += verdict
            except Exception as exc:  # noqa: BLE001 — one flaky item shouldn't kill the run
                ans = "!"
                print(f"   ⚠ gen skipped for {it['id']} ({type(exc).__name__})")
            time.sleep(GEN_THROTTLE_S)

        print(f"{it['id']:<22}{it['type']:<13}{retr:<6}{rr.top_cosine:<7.3f}{gate:<6}{ans:<5}")

    n_exp = sum(1 for it in items if it.get("expected_sources"))
    n_oos = sum(1 for it in items if it["type"] == "out-of-scope")
    print("\n" + "=" * 64)
    print("📊 RESULTS")
    print(f"  Retrieval hit-rate@{settings.final_k}:  {hits}/{n_exp} = {hits / max(n_exp,1):.0%}")
    print(f"  Gate accuracy (out-of-scope): {gate_ok}/{n_oos} = {gate_ok / max(n_oos,1):.0%}")
    print(f"  Mean top cosine:              {sum(cosines)/max(len(cosines),1):.3f}")
    if do_gen:
        print(f"  Answer fact match:            {ans_ok}/{ans_total} = {ans_ok / max(ans_total,1):.0%}")
        print(f"  Faithfulness (LLM-judge):     {faithful}/{faithful_total} = {faithful / max(faithful_total,1):.0%}")
    print("=" * 64)


if __name__ == "__main__":
    main()
