"""Quick CLI to query the finance RAG without starting the server.

    python scripts/ask.py "What are NVIDIA's main risk factors?"
    python scripts/ask.py            # interactive loop

Prints the answer plus diagnostics (gated?, top cosine, metadata filters,
citations) so you can see retrieval quality at a glance.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.pipeline import build_pipeline  # noqa: E402
from config import settings  # noqa: E402


def _show(result: dict) -> None:
    print("\n" + result["answer"])
    flag = "⚠ NOT FOUND / gated" if result["gated"] else "✓ grounded"
    filt = result.get("filters") or {}
    filt_str = ""
    if filt.get("tickers") or filt.get("years"):
        filt_str = f" filters={filt.get('tickers', [])}/{filt.get('years', [])}"
    print(f"\n— [{flag}] top_cosine={result['top_cosine']} latency={result['latency_ms']}ms{filt_str}")
    if result["sources"]:
        print("  sources: " + ", ".join(f"{s['citation']} ({s['score']})" for s in result["sources"]))


def main() -> None:
    pipeline = build_pipeline(settings)
    if len(sys.argv) > 1:
        _show(pipeline.answer(" ".join(sys.argv[1:])))
        return
    print("FinSight — ask about NVDA / AAPL / MSFT 10-K (Ctrl+C to quit).")
    try:
        while True:
            q = input("\n> ").strip()
            if q:
                _show(pipeline.answer(q))
    except (KeyboardInterrupt, EOFError):
        print("\nbye 👋")


if __name__ == "__main__":
    main()
