"""Central configuration for the finance-RAG backend.

All knobs live here and are overridable via environment variables / a local
``.env`` file (see ``.env.example``). Paths resolve relative to the project root
so the app runs the same from any working directory.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Gemini ---------------------------------------------------------
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    generation_model: str = Field(default="gemini-2.5-flash", alias="GENERATION_MODEL")
    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=768, alias="EMBEDDING_DIM")

    # --- Paths ----------------------------------------------------------
    corpus_path: Path = Field(default=PROJECT_ROOT / "data" / "corpus", alias="CORPUS_PATH")
    index_path: Path = Field(default=PROJECT_ROOT / "data" / "index", alias="INDEX_PATH")
    log_path: Path = Field(default=PROJECT_ROOT / "logs" / "queries.jsonl", alias="LOG_PATH")

    # --- EDGAR fetch ----------------------------------------------------
    # SEC requires a descriptive User-Agent with contact info on every request.
    edgar_user_agent: str = Field(
        default="Nehemiah finance-rag nehemiahwj@gmail.com", alias="EDGAR_USER_AGENT"
    )
    edgar_tickers: str = Field(default="NVDA,AAPL,MSFT", alias="EDGAR_TICKERS")

    # --- Chunking (10-K filings are large) ------------------------------
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # --- Retrieval ------------------------------------------------------
    top_k_dense: int = Field(default=10, alias="TOP_K_DENSE")
    top_k_sparse: int = Field(default=10, alias="TOP_K_SPARSE")
    rrf_k: int = Field(default=60, alias="RRF_K")  # Reciprocal Rank Fusion constant
    final_k: int = Field(default=6, alias="FINAL_K")
    # Below this top cosine similarity, retrieval is "weak" -> gated (answer
    # "not found in filings" instead of risking a fabricated financial fact).
    confidence_threshold: float = Field(default=0.62, alias="CONFIDENCE_THRESHOLD")

    # --- Reranker (optional, needs sentence-transformers) ---------------
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    # --- Vector store backend ------------------------------------------
    vector_store: str = Field(default="local", alias="VECTOR_STORE")  # local | supabase
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_key: str = Field(default="", alias="SUPABASE_KEY")
    supabase_table: str = Field(default="filing_chunks", alias="SUPABASE_TABLE")

    @property
    def tickers(self) -> list[str]:
        return [t.strip().upper() for t in self.edgar_tickers.split(",") if t.strip()]


settings = Settings()
