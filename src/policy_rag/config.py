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
    gemini_api_key: str
    gemini_model: str
    gemini_rpm: int
    groq_api_key: str
    groq_model: str
    groq_rpm: int


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
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        # The free tier allows 5 requests per minute per model.
        gemini_rpm=int(os.getenv("GEMINI_RPM", "5")),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        # The free tier allows 30 requests per minute (and 8,000 tokens per minute).
        groq_rpm=int(os.getenv("GROQ_RPM", "30")),
    )
