"""FastAPI app exposing the finance RAG chatbot, and serving its web client.

Endpoints:
  GET  /health        -> liveness + index status + whether the pipeline really built
  POST /chat          -> blocking JSON answer (with sources + diagnostics)
  POST /chat/stream   -> Server-Sent Events: sources first, then tokens, then done
  GET  /              -> the web client in ``web/``

The UI is mounted on the same origin as the API on purpose. It used to be a loose
file you opened from the filesystem, which meant a demo was "start uvicorn, then go
find web/index.html in a file manager", and the page had to hardcode a localhost URL
to talk back. One origin removes both problems, and ``python demo.py`` opens it.

The pipeline is built once at startup and cached. CORS stays open for local dev; lock
it down to your domain before deploying.
"""
from __future__ import annotations

import json
import re
import sys
from contextlib import asynccontextmanager

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.observability import log_query
from app.rag.generator import describe_failure
from app.schemas import ChatRequest, ChatResponse
from config import PROJECT_ROOT, settings

WEB_DIR = PROJECT_ROOT / "web"

_pipeline = None
_pipeline_error: str | None = None

# Errors that reach the browser are raw exception text, and the query embedder talks
# to Gemini too, so a failure there can carry a request URL. Redact anything shaped
# like a key before it leaves the process.
_SECRET_RE = re.compile(r"(AIza[0-9A-Za-z_\-]{10,})|((?i:key|token)=)[^\s&\"']+")


def _safe(message: object) -> str:
    return _SECRET_RE.sub(lambda m: (m.group(2) or "") + "REDACTED", str(message))


def _explain(exc: BaseException) -> str:
    """One readable line for the browser.

    Retrieval embeds the query through the same Gemini account as generation, so a
    quota failure surfaces here as several kilobytes of SDK JSON. Reuse the
    generator's classifier so the banner says what happened, and redact regardless
    in case the text falls through to the generic branch.
    """
    return _safe(describe_failure(exc))


def get_pipeline():
    """Return the cached pipeline, building it on first use."""
    global _pipeline, _pipeline_error
    if _pipeline is None:
        from app.rag.pipeline import build_pipeline

        try:
            _pipeline = build_pipeline(settings)
            _pipeline_error = None
        except Exception as exc:
            _pipeline_error = _safe(exc)
            raise
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the pipeline so the first question is not the one that pays for loading
    # the index. A failure here is reported by /health rather than killing the app,
    # because a running server that can explain what is wrong beats a dead one.
    try:
        get_pipeline()
        print("[ok] RAG pipeline ready.")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] pipeline not ready: {exc}")
    yield


app = FastAPI(title="FinSight - SEC 10-K RAG", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> JSONResponse:
    """Report what is actually true, including a 503 when the app cannot answer.

    This used to return ``{"status": "ok"}`` unconditionally, even when the pipeline
    had failed to build and every question would 503. A health check that cannot go
    unhealthy is decoration.
    """
    index_ready = (settings.index_path / "chunks.json").exists()
    pipeline_ready = _pipeline is not None
    ok = index_ready and pipeline_ready
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ok" if ok else "degraded",
            "index_ready": index_ready,
            "pipeline_ready": pipeline_ready,
            "error": _pipeline_error,
            "model": settings.generation_model,
        },
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        history = [m.model_dump() for m in req.history]
        result = get_pipeline().answer(req.message, history)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=_safe(exc))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # Generation failures are already handled inside the pipeline, so reaching
        # here means retrieval or index loading broke. Say so instead of a bare 500.
        raise HTTPException(status_code=503, detail=f"retrieval unavailable: {_explain(exc)}")
    log_query(settings.log_path, query=req.message, result=result)
    return ChatResponse(**result)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    history = [m.model_dump() for m in req.history]

    def event_stream():
        captured = {
            "sources": [], "gated": None, "top_cosine": None,
            "filters": {}, "latency_ms": None, "degraded": False, "notice": None,
        }
        try:
            for event in get_pipeline().stream_answer(req.message, history):
                if event["type"] == "sources":
                    captured.update(event)
                elif event["type"] == "error":
                    captured["degraded"] = True
                    captured["notice"] = event["message"]
                elif event["type"] == "done":
                    captured["latency_ms"] = event["latency_ms"]
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            # The response is already a 200 with headers sent, so an exception here
            # cannot become an HTTP error. Without this the stream simply stops and
            # the page sits on its citations forever, which is the failure mode that
            # is most confusing to watch. Send a terminal error event instead.
            captured["degraded"] = True
            captured["notice"] = _explain(exc)
            for event in (
                {
                    "type": "error",
                    "message": f"retrieval unavailable: {_explain(exc)}",
                    "recovered": False,
                },
                {"type": "done", "latency_ms": None},
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        log_query(
            settings.log_path,
            query=req.message,
            result={
                "gated": captured["gated"],
                "top_cosine": captured["top_cosine"],
                "filters": captured.get("filters", {}),
                "sources": captured["sources"],
                "latency_ms": captured["latency_ms"],
                "degraded": captured["degraded"],
                "notice": captured["notice"],
            },
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Mounted last so every API route above wins the match. `html=True` serves
# web/index.html at `/`.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
