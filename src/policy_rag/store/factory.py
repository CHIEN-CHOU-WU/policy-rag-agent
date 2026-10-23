"""Builds the configured stores. The one place that picks storage adapters."""

from policy_rag.config import Settings
from policy_rag.store.base import VectorStore
from policy_rag.store.page_store import JsonPageStore, PageStore


def create_vector_store(settings: Settings) -> VectorStore:
    # Imported here so callers never load an adapter's vendor SDK by importing this module.
    from policy_rag.store.chroma_store import ChromaVectorStore

    return ChromaVectorStore(settings.chroma_dir, settings.collection_name)


def create_page_store(settings: Settings) -> PageStore:
    return JsonPageStore(settings.pages_path)
