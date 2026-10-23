"""Runtime settings loaded from environment variables (and an optional ``.env`` file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    raw_data_dir: Path
    chroma_dir: Path
    collection_name: str


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        raw_data_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", "chroma_db")),
        collection_name=os.getenv("COLLECTION_NAME", "policy_chunks"),
    )
