"""Runtime settings, read from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # "anthropic" (Claude) or "openrouter" (incl. free ':free' models).
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "anthropic").lower()
    claude_model: str = os.getenv("RAG_MODEL", "claude-opus-5-5")
    openrouter_model: str = os.getenv("RAG_OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "./data.nosync"))
    # Approach 1: fixed-size chunks of ~500-1000 tokens with 10-20% overlap.
    chunk_tokens: int = int(os.getenv("RAG_CHUNK_TOKENS", "700"))
    chunk_overlap: float = float(os.getenv("RAG_CHUNK_OVERLAP", "0.15"))
    # Approach 1: cosine top-K with K = 4-8.
    top_k: int = int(os.getenv("RAG_TOP_K", "6"))
    collection: str = os.getenv("RAG_COLLECTION", "documents")
    max_upload_mb: int = int(os.getenv("RAG_MAX_UPLOAD_MB", "50"))

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


settings = Settings()
