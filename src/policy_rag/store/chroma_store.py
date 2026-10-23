"""Chroma adapter for the ``VectorStore`` port.

This is the only module that imports ``chromadb``. Embeddings use
all-MiniLM-L6-v2 run locally via ONNX, so indexing needs no API key.
"""

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from policy_rag.models import Chunk, SearchResult
from policy_rag.store.base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, path: Path, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection_name = collection_name
        self._opened: chromadb.Collection | None = None

    def add(self, chunks: Sequence[Chunk]) -> None:
        batch_size = self._client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[_to_metadata(chunk) for chunk in batch],
            )

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        result = self._collection.query(query_texts=[query], n_results=k)
        return [
            SearchResult(
                chunk=_from_record(chunk_id, document, metadata),
                score=1.0 - distance,
            )
            for chunk_id, document, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        # Deleting does not open the old collection, so this also recovers an
        # index built with a different embedding model.
        try:
            self._client.delete_collection(self._collection_name)
        except NotFoundError:
            pass
        self._opened = None

    @property
    def _collection(self) -> chromadb.Collection:
        """The collection, opened (or created) on first use."""
        if self._opened is None:
            try:
                self._opened = self._client.get_or_create_collection(
                    self._collection_name,
                    # Pinned explicitly: indexing and querying must use the same model,
                    # so a change in Chroma's default must not silently switch it.
                    embedding_function=ONNXMiniLM_L6_V2(),
                    configuration={"hnsw": {"space": "cosine"}},
                )
            except ValueError as error:
                raise RuntimeError(
                    f"The index in {self._client.get_settings().persist_directory}/ was built "
                    "with a different embedding model. Rebuild it with "
                    "`uv run python -m policy_rag.ingest`."
                ) from error
        return self._opened


def _to_metadata(chunk: Chunk) -> dict[str, Any]:
    metadata = asdict(chunk)
    del metadata["text"]
    return metadata


def _from_record(chunk_id: str, document: str, metadata: dict[str, Any]) -> Chunk:
    fields = {**metadata, "chunk_id": chunk_id, "text": document}
    return Chunk(**fields)
