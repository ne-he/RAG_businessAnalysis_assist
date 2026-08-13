"""What happens when Gemini is not there (offline, no key, no network).

The free tier rate-limits generation to a few requests per minute, so a 429 during a
demo is routine. The thing being pinned here is that a failure is visible and the
retrieved evidence survives it. Before this, a mid-stream failure produced a page
showing citations and then nothing at all, forever: `sources` is emitted before the
first token, and the client only handled a failed `fetch`, not a stream that died
after a successful one.
"""
from __future__ import annotations

import pytest

from app.main import _explain, _safe
from app.rag.generator import (
    GenerationFailed,
    _call_with_retry,
    _is_retryable,
    describe_failure,
)
from app.rag.pipeline import (
    FALLBACK_NOTICE,
    RAGPipeline,
    _extractive_answer,
    _mark_partial_lead,
)
from app.rag.retriever import RetrievalResult
from tests.conftest import CORPUS, make_hit


# ---- retry classification ------------------------------------------------

@pytest.mark.parametrize("message", [
    "429 RESOURCE_EXHAUSTED: quota exceeded",
    "503 Service Unavailable",
    "Deadline exceeded",
    "rate limit reached for model",
])
def test_transient_failures_are_retryable(message):
    assert _is_retryable(RuntimeError(message)) is True


@pytest.mark.parametrize("message", [
    "400 INVALID_ARGUMENT: contents must not be empty",
    "API key not valid",
])
def test_permanent_failures_are_not_retryable(message):
    assert _is_retryable(RuntimeError(message)) is False


def test_retry_succeeds_on_the_second_attempt(monkeypatch):
    monkeypatch.setattr("app.rag.generator._RETRY_DELAY_SEC", 0)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "recovered"

    assert _call_with_retry(flaky) == "recovered"
    assert len(calls) == 2


def test_permanent_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr("app.rag.generator._RETRY_DELAY_SEC", 0)
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("API key not valid")

    with pytest.raises(GenerationFailed):
        _call_with_retry(broken)
    assert len(calls) == 1, "a permanent error must not burn a second request"


def test_failure_reason_does_not_leak_the_raw_payload():
    reason = describe_failure(RuntimeError("429 RESOURCE_EXHAUSTED key=AIzaSyTOPSECRET quota"))
    assert "AIzaSy" not in reason
    assert "rate limit" in reason.lower()


# ---- nothing secret reaches the browser ----------------------------------

@pytest.mark.parametrize("raw", [
    "connect failed: https://api.example.com/v1?key=AIzaSyABCDEFGHIJ1234567890",
    "auth error for AIzaSyABCDEFGHIJ1234567890",
    "bad token=ghp_sometokenvalue12345",
])
def test_error_text_sent_to_the_client_is_redacted(raw):
    """Retrieval embeds the query through Gemini too, so its errors can carry a key."""
    cleaned = _safe(RuntimeError(raw))
    assert "AIzaSy" not in cleaned
    assert "ghp_sometokenvalue" not in cleaned
    assert "REDACTED" in cleaned


def test_redaction_keeps_the_message_readable():
    cleaned = _safe(RuntimeError("connect failed: timeout after 30s"))
    assert cleaned == "connect failed: timeout after 30s"


def test_quota_error_reaches_the_browser_as_one_readable_line():
    """The SDK's 429 body is several KB of JSON. Nobody should see that in a demo."""
    raw = RuntimeError(
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
        "current quota... Quota exceeded for metric: "
        "generativelanguage.googleapis.com/embed_content_free_tier_requests, limit: 1000'}}"
    )
    explained = _explain(raw)
    assert explained == (
        "Gemini rate limit reached (free tier allows only a few requests per minute)"
    )
    assert len(explained) < 120


# ---- extractive fallback -------------------------------------------------

def _retrieval(gated: bool = False) -> RetrievalResult:
    hits = [make_hit(*c) for c in CORPUS]
    for i, hit in enumerate(hits):
        hit.score = 1.0 - i * 0.1
    return RetrievalResult(
        hits=[] if gated else hits,
        gated=gated,
        top_cosine=0.2 if gated else 0.8,
        filters={"tickers": [], "years": []},
    )


def test_extractive_answer_keeps_the_citations():
    text = _extractive_answer(_retrieval())
    assert FALLBACK_NOTICE in text
    assert "[NVDA FY2024" in text, "the fallback must stay attributable"


def test_extractive_answer_respects_the_gate():
    """A gated query must not start quoting filings at the user anyway."""
    text = _extractive_answer(_retrieval(gated=True))
    assert FALLBACK_NOTICE not in text
    assert "could not find support" in text


def test_extractive_answer_deduplicates_sections():
    """One passage per (ticker, year, section), not four near-identical ones."""
    hits = [make_hit(*CORPUS[0]) for _ in range(4)]
    result = RetrievalResult(hits=hits, gated=False, top_cosine=0.9, filters={})
    assert _extractive_answer(result).count("[NVDA FY2024") == 1


def test_passage_starting_mid_sentence_is_marked_not_trimmed():
    """The overlap tail means chunks routinely start mid-word."""
    text = "elines, and these may not deliver benefits. We also offer software solutions."
    marked = _mark_partial_lead(text)
    assert marked.startswith("... ")
    assert text in marked, "quoting a filing must never silently drop its opening"


def test_text_already_starting_cleanly_is_untouched():
    assert _mark_partial_lead("Competition is intense.") == "Competition is intense."
    assert _mark_partial_lead("2026 revenue rose.") == "2026 revenue rose."


# ---- pipeline degradation ------------------------------------------------

class _DeadGenerator:
    """Fails the way a rate-limited Gemini does: before producing anything."""

    def generate(self, query, retrieval, history=None):
        raise GenerationFailed("Gemini rate limit reached")

    def stream(self, query, retrieval, history=None):
        raise GenerationFailed("Gemini rate limit reached")
        yield  # pragma: no cover - makes this a generator function


class _DyingGenerator:
    """Fails partway through, which is the case a retry cannot rescue."""

    def generate(self, query, retrieval, history=None):
        raise GenerationFailed("Gemini rate limit reached")

    def stream(self, query, retrieval, history=None):
        yield "NVIDIA's main risks "
        raise GenerationFailed("Gemini rate limit reached")


class _StubRetriever:
    def __init__(self, result):
        self._result = result

    def retrieve(self, query):
        return self._result


def test_blocking_answer_degrades_instead_of_raising():
    pipeline = RAGPipeline(_StubRetriever(_retrieval()), _DeadGenerator())
    result = pipeline.answer("What are NVIDIA's risk factors?")

    assert result["degraded"] is True
    assert result["notice"] == "Gemini rate limit reached"
    assert FALLBACK_NOTICE in result["answer"]
    assert result["sources"], "retrieval succeeded, so the sources must survive"
    assert result["top_cosine"] == 0.8


def test_healthy_answer_is_not_marked_degraded():
    class _Working:
        def generate(self, query, retrieval, history=None):
            return "NVIDIA cites competition and supply concentration."

    pipeline = RAGPipeline(_StubRetriever(_retrieval()), _Working())
    result = pipeline.answer("q")
    assert result["degraded"] is False
    assert result["notice"] is None


def test_stream_emits_an_error_event_then_a_usable_fallback():
    pipeline = RAGPipeline(_StubRetriever(_retrieval()), _DeadGenerator())
    events = list(pipeline.stream_answer("What are NVIDIA's risk factors?"))
    kinds = [e["type"] for e in events]

    assert kinds[0] == "sources", "the SSE contract puts sources first"
    assert "error" in kinds, "a silent stop is the bug this test exists for"
    assert kinds[-1] == "done", "the client waits on `done` to re-enable the input"

    error = events[kinds.index("error")]
    assert error["recovered"] is True, "nothing had been shown yet, so this is recoverable"

    body = "".join(e["text"] for e in events if e["type"] == "token")
    assert FALLBACK_NOTICE in body
    assert "[NVDA FY2024" in body


def test_stream_failing_midway_is_reported_as_unrecovered():
    """Tokens are already on screen, so the fallback would duplicate them."""
    pipeline = RAGPipeline(_StubRetriever(_retrieval()), _DyingGenerator())
    events = list(pipeline.stream_answer("q"))

    error = next(e for e in events if e["type"] == "error")
    assert error["recovered"] is False

    body = "".join(e["text"] for e in events if e["type"] == "token")
    assert body == "NVIDIA's main risks ", "partial text must not be topped up with a fallback"
    assert [e["type"] for e in events][-1] == "done"
