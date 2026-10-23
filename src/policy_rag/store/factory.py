"""Builds the configured ``VectorStore``. The one place that picks an adapter."""

from policy_rag.config import Settings
from policy_rag.store.base import VectorStore


def create_vector_store(settings: Settings) -> VectorStore:
    # Imported here so callers never load an adapter's vendor SDK by importing this module.
    from policy_rag.store.chroma_store import ChromaVectorStore

    return ChromaVectorStore(settings.chroma_dir, settings.collection_name)
