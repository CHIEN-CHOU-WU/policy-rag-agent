"""Retrieve relevant chunks and format them as context for a language model."""

from policy_rag.models import SearchResult
from policy_rag.store.base import VectorStore


def retrieve(store: VectorStore, question: str, k: int = 5) -> list[SearchResult]:
    return store.search(question, k=k)


def format_context(results: list[SearchResult]) -> str:
    """Wrap each chunk in an ``<excerpt>`` tag that carries its page and section.

    The page attribute is exactly what the model is asked to cite, so every
    citation can be checked against the excerpts it was given.
    """
    excerpts = []
    for result in results:
        chunk = result.chunk
        section = chunk.section_title.replace('"', "'")
        excerpts.append(
            f'<excerpt page="{chunk.page_label}" section="{section}">\n{chunk.body}\n</excerpt>'
        )
    return "\n\n".join(excerpts)
