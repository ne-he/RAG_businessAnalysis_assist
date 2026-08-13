"""Generation layer (Gemini, ``google-genai`` SDK) with streaming + history.

Loads the financial-analyst system prompt once into the model config, then
generates grounded answers from the retrieved context. Supports both a blocking
``generate`` and a token-streaming ``stream`` for a responsive chat UI.

Every call into Gemini is wrapped. The free tier rate-limits generation to a
handful of requests per minute, so a 429 in the middle of a demo is a routine
event, not an exotic one. Transient failures are retried once, and anything that
survives that is re-raised as ``GenerationFailed`` carrying a short, safe reason,
which is what the pipeline needs in order to fall back instead of dying.

Streaming needs one extra care. ``generate_content_stream`` is lazy, so a 429
surfaces on the first read rather than at the call, and the retry has to cover
that first read too. Once a token has reached the client a retry is no longer
possible: replaying would duplicate text on screen. So the retry window closes
after the first chunk, and later failures are reported instead of retried.
"""
from __future__ import annotations

import time
from typing import Iterator

from app.rag.prompts import build_user_turn, load_system_prompt
from app.rag.retriever import RetrievalResult

# Substrings that mark a failure worth retrying. Matched against the string form
# of the exception because the SDK raises several unrelated error classes for
# what is operationally the same "try again shortly".
_RETRYABLE = (
    "429", "resource_exhausted", "rate limit", "quota",
    "503", "unavailable", "500", "internal", "deadline", "timeout",
)

_RETRY_ATTEMPTS = 2
_RETRY_DELAY_SEC = 2.0


class GenerationFailed(RuntimeError):
    """Gemini could not produce an answer. The message is safe to show a user."""


def _is_retryable(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE)


def describe_failure(exc: BaseException) -> str:
    """A short, non-leaking explanation. Never includes the API key or raw payload.

    Shared with the API layer: retrieval embeds the query through the same Gemini
    account, so it hits the same quota and deserves the same one-line explanation
    rather than the SDK's several-kilobyte JSON error body.
    """
    text = f"{exc}".lower()
    if "429" in text or "resource_exhausted" in text or "quota" in text or "rate limit" in text:
        return "Gemini rate limit reached (free tier allows only a few requests per minute)"
    if "api key" in text or "unauthenticated" in text or "permission" in text:
        return "Gemini rejected the API key"
    if "not found" in text or "404" in text:
        return "the configured Gemini model is unavailable"
    if any(m in text for m in ("timeout", "deadline", "connection", "network")):
        return "could not reach Gemini (network)"
    return "the Gemini request failed"


def _call_with_retry(fn):
    """Run ``fn``, retrying once on a transient error, then raise GenerationFailed."""
    last: BaseException | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised as GenerationFailed below
            last = exc
            if attempt == _RETRY_ATTEMPTS - 1 or not _is_retryable(exc):
                break
            time.sleep(_RETRY_DELAY_SEC)
    raise GenerationFailed(describe_failure(last)) from last


class Generator:
    def __init__(self, api_key: str, model_name: str):
        from google import genai  # lazy
        from google.genai import types

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is empty. Set it in .env.")
        self._client = genai.Client(api_key=api_key)
        self._model = model_name
        self._config = types.GenerateContentConfig(
            system_instruction=load_system_prompt()
        )

    @staticmethod
    def _to_contents(history: list[dict] | None, user_turn: str) -> list[dict]:
        contents: list[dict] = []
        for msg in history or []:
            role = "model" if msg.get("role") in ("assistant", "model") else "user"
            contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
        contents.append({"role": "user", "parts": [{"text": user_turn}]})
        return contents

    def generate(self, query: str, retrieval: RetrievalResult, history=None) -> str:
        user_turn = build_user_turn(query, retrieval.hits, retrieval.gated)
        contents = self._to_contents(history, user_turn)

        def call():
            resp = self._client.models.generate_content(
                model=self._model, contents=contents, config=self._config
            )
            return resp.text

        text = _call_with_retry(call)
        if not text:
            raise GenerationFailed("Gemini returned an empty answer")
        return text

    def stream(self, query: str, retrieval: RetrievalResult, history=None) -> Iterator[str]:
        user_turn = build_user_turn(query, retrieval.hits, retrieval.gated)
        contents = self._to_contents(history, user_turn)

        def open_stream():
            # Pull the first chunk inside the retry window: the SDK is lazy, so a
            # 429 shows up here rather than on the call above, and this is the last
            # moment a retry is still invisible to the client.
            chunks = iter(
                self._client.models.generate_content_stream(
                    model=self._model, contents=contents, config=self._config
                )
            )
            return next(chunks, None), chunks

        first, rest = _call_with_retry(open_stream)

        if first is not None and getattr(first, "text", None):
            yield first.text

        while True:
            try:
                chunk = next(rest)
            except StopIteration:
                break
            except Exception as exc:  # noqa: BLE001
                # Past the retry window: tokens are already on screen, and replaying
                # the request would duplicate them. Report instead.
                raise GenerationFailed(describe_failure(exc)) from exc
            if getattr(chunk, "text", None):
                yield chunk.text
