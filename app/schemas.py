"""Pydantic request/response models for the HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[Message] = Field(default_factory=list)


class Source(BaseModel):
    ticker: str
    fiscal_year: str
    section: str
    source_url: str
    citation: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    gated: bool
    top_cosine: float
    filters: dict = Field(default_factory=dict)
    latency_ms: float
