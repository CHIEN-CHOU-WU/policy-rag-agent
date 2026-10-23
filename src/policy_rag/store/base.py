"""The vector store port: what the rest of the application needs from storage.

Code outside ``policy_rag.store`` depends only on this interface. Concrete
adapters (Chroma today, pgvector later) own embedding and persistence.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from policy_rag.models import Chunk, SearchResult


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: Sequence[Chunk]) -> None:
        """Embed and store chunks. Re-adding a ``chunk_id`` replaces it."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the ``k`` chunks most similar to ``query``, best first."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored chunks."""

    @abstractmethod
    def reset(self) -> None:
        """Remove every stored chunk."""
