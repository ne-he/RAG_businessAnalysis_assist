"""One command to demo FinSight: `python demo.py`.

Starts the API, serves the web client from the same origin, and opens a browser at
it. Replaces the old three-step ritual of starting uvicorn in a terminal, finding
`web/index.html` in a file manager, and opening it by hand, which is a bad thing to
be doing while someone is watching.

Preflight runs before the server binds, because the two ways this demo goes wrong
are both silent and both look identical from the browser: an index that was never
built, and a missing API key. Better to say so in the terminal than to let someone
type a question and watch nothing happen.
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser

from config import settings

DEFAULT_PORT = 8000


def _index_ready() -> bool:
    return (settings.index_path / "chunks.json").exists()


def _free_port(preferred: int) -> int:
    """Return `preferred`, or the next free port, so a stale server is not fatal."""
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def preflight() -> bool:
    """Print what is missing. Returns False only for things that make the demo useless."""
    ok = True

    if _index_ready():
        print(f"[ok]   index found at {settings.index_path}")
    else:
        ok = False
        print(f"[FAIL] no index at {settings.index_path}")
        print("       Build it once (needs a Gemini key and ~10 minutes):")
        print("         python scripts/fetch_edgar.py")
        print("         python scripts/ingest.py")

    if settings.gemini_api_key:
        print(f"[ok]   Gemini key set, model {settings.generation_model}")
    else:
        # Not fatal on purpose: retrieval still works and the pipeline falls back to
        # quoting the filings, so the demo degrades instead of dying.
        print("[warn] GEMINI_API_KEY is empty. Retrieval and citations still work,")
        print("       but answers will be quoted from the filings, not written.")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FinSight demo.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args()

    print("FinSight demo")
    print("-" * 52)
    if not preflight():
        print("-" * 52)
        print("Refusing to start: there is no index to answer from.")
        return 1

    port = _free_port(args.port)
    if port != args.port:
        print(f"[warn] port {args.port} is busy, using {port}")

    url = f"http://127.0.0.1:{port}"
    print("-" * 52)
    print(f"Opening {url}   (Ctrl+C to stop)")

    if not args.no_browser:
        # Fires after uvicorn has had a moment to bind; opening earlier races the
        # server and shows a connection error page.
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
