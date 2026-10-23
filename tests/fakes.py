"""In-memory stand-ins for the ports, so tests never touch a model, a database, or the network."""

from collections.abc import Sequence

from policy_rag.llm.base import AssistantTurn, LLMClient, Message, ToolSpec, UserMessage
from policy_rag.models import Chunk, PageText, SearchResult
from policy_rag.store.base import VectorStore
from policy_rag.store.page_store import PageStore


def make_chunk(chunk_id: str, section: str, body: str, page_start: int, page_end: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=f"{section}\n\n{body}",
        source="handbook.pdf",
        section_title=section,
        section_path=section,
        page_start=page_start,
        page_end=page_end,
        pdf_page_start=page_start,
        pdf_page_end=page_end,
        page_numbering="printed",
        content_type="text",
    )


def make_page(page_number: int, text: str) -> PageText:
    return PageText(
        source="handbook.pdf",
        page_number=page_number,
        pdf_page=page_number,
        page_numbering="printed",
        text=text,
    )


class FakeStore(VectorStore):
    """Returns fixed chunks; ``by_query`` overrides the results for specific queries."""

    def __init__(self, chunks: list[Chunk], by_query: dict[str, list[Chunk]] | None = None) -> None:
        self.chunks = chunks
        self.by_query = by_query or {}
        self.queries: list[str] = []

    def add(self, chunks: Sequence[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self.queries.append(query)
        chunks = self.by_query.get(query, self.chunks)
        return [SearchResult(chunk, score=1.0) for chunk in chunks[:k]]

    def count(self) -> int:
        return len(self.chunks)

    def reset(self) -> None:
        raise NotImplementedError


class FakePageStore(PageStore):
    def __init__(self, pages: list[PageText]) -> None:
        self.pages = pages

    def save(self, pages: Sequence[PageText]) -> None:
        self.pages = list(pages)

    def get(self, page_number: int) -> list[PageText]:
        return [page for page in self.pages if page.page_number == page_number]


class FakeLLM(LLMClient):
    """Always replies with the same text and records each (system, prompt) it was given."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        prompt = next(m.text for m in reversed(messages) if isinstance(m, UserMessage))
        self.calls.append((system, prompt))
        return AssistantTurn(text=self.reply)


class ScriptedLLM(LLMClient):
    """Plays back a fixed sequence of turns and records the conversation it saw each time."""

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.turns = list(turns)
        self.seen: list[list[Message]] = []

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        self.seen.append(list(messages))
        if not self.turns:
            raise AssertionError("ScriptedLLM ran out of turns")
        return self.turns.pop(0)
