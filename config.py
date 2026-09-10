"""Central configuration. All settings read from .env via python-dotenv."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def ensure_api_key() -> str | None:
    """Return the OpenRouter key from env, or prompt if running in a TTY."""
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    if not sys.stdin.isatty():
        return None
    try:
        from getpass import getpass
        key = getpass("Enter your OpenRouter API key (hidden): ").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    return key or None


@dataclass
class Config:
    # Paths
    CORPUS_DIR: Path = Path("data/corpus")
    INDEX_DIR: Path = Path("index")

    # Retrieval
    TOP_K: int = 5
    TOP_N_RERANK: int = 20
    MAX_CONTEXT_CHUNKS: int = 10

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # Local models
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBED_BATCH_SIZE: int = 32

    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    LLM_TEMPERATURE: float = 0.0
    OPENROUTER_API_KEY: str | None = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    OPENROUTER_BASE_URL: str | None = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
    )

    def resolved_index_dir(self) -> Path:
        return self.INDEX_DIR.resolve()

    def resolved_corpus_dir(self) -> Path:
        return self.CORPUS_DIR.resolve()


CONFIG = Config()