"""Generation layer (Gemini, ``google-genai`` SDK) with streaming + history.

Loads the financial-analyst system prompt once into the model config, then
generates grounded answers from the retrieved context. Supports both a blocking
``generate`` and a token-streaming ``stream`` for a responsive chat UI.
"""
from __future__ import annotations

from typing import Iterator

from app.rag.prompts import build_user_turn, load_system_prompt
from app.rag.retriever import RetrievalResult


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
        resp = self._client.models.generate_content(
            model=self._model,
            contents=self._to_contents(history, user_turn),
            config=self._config,
        )
        return resp.text

    def stream(self, query: str, retrieval: RetrievalResult, history=None) -> Iterator[str]:
        user_turn = build_user_turn(query, retrieval.hits, retrieval.gated)
        stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=self._to_contents(history, user_turn),
            config=self._config,
        )
        for chunk in stream:
            if getattr(chunk, "text", None):
                yield chunk.text
