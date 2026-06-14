"""FastAPI app exposing the finance RAG chatbot.

Endpoints:
  GET  /health        -> liveness + index status
  POST /chat          -> blocking JSON answer (with sources + diagnostics)
  POST /chat/stream   -> Server-Sent Events: sources first, then tokens, then done

The pipeline is built once at startup. CORS is open for local dev; lock it down
to your domain before deploying.
"""
from __future__ import annotations

import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.observability import log_query
from app.schemas import ChatRequest, ChatResponse
from config import settings

app = FastAPI(title="FinSight — SEC 10-K RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from app.rag.pipeline import build_pipeline

        _pipeline = build_pipeline(settings)
    return _pipeline


@app.on_event("startup")
def _startup() -> None:
    try:
        get_pipeline()
        print("✅ RAG pipeline ready.")
    except Exception as exc:  # surfaced on first request too
        print(f"⚠  pipeline not ready: {exc}")


@app.get("/health")
def health() -> dict:
    ready = (settings.index_path / "chunks.json").exists()
    return {"status": "ok", "index_ready": ready, "model": settings.generation_model}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        history = [m.model_dump() for m in req.history]
        result = get_pipeline().answer(req.message, history)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    log_query(settings.log_path, query=req.message, result=result)
    return ChatResponse(**result)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    history = [m.model_dump() for m in req.history]

    def event_stream():
        captured = {"sources": [], "gated": None, "top_cosine": None, "filters": {}, "latency_ms": None}
        for event in get_pipeline().stream_answer(req.message, history):
            if event["type"] == "sources":
                captured.update(event)
            elif event["type"] == "done":
                captured["latency_ms"] = event["latency_ms"]
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
            },
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
