"""Runtime settings loaded from environment variables (and an optional ``.env`` file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    raw_data_dir: Path
    chroma_dir: Path
    pages_path: Path
    collection_name: str
    llm_provider: str
    anthropic_model: str
    ollama_model: str
    ollama_host: str


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        raw_data_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", "chroma_db")),
        pages_path=Path(os.getenv("PAGES_PATH", "data/processed/pages.json")),
        collection_name=os.getenv("COLLECTION_NAME", "policy_chunks"),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )
