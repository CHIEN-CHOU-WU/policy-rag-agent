from collections.abc import Sequence

import pytest

from policy_rag.llm.base import LLMClient
from policy_rag.models import Chunk, SearchResult
from policy_rag.rag import NOT_FOUND, SYSTEM_PROMPT, answer_question, cited_pages
from policy_rag.retrieve import format_context
from policy_rag.store.base import VectorStore


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


PART_B = make_chunk("c1", "Part B", "Part B covers outpatient care.", 32, 32)
ENROLLMENT = make_chunk("c2", "Signing up", "You can sign up when you turn 65.", 17, 18)


class FakeStore(VectorStore):
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def add(self, chunks: Sequence[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return [SearchResult(chunk, score=1.0) for chunk in self.chunks[:k]]

    def count(self) -> int:
        return len(self.chunks)

    def reset(self) -> None:
        raise NotImplementedError


class FakeLLM(LLMClient):
    """Returns a canned reply and records what it was asked."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.reply


def test_prompt_contains_each_excerpt_with_its_page() -> None:
    llm = FakeLLM("Outpatient care [p. 32].")
    answer_question("What does Part B cover?", FakeStore([PART_B, ENROLLMENT]), llm)

    [(system, prompt)] = llm.calls
    assert system == SYSTEM_PROMPT
    assert '<excerpt page="32" section="Part B">\nPart B covers outpatient care.\n</excerpt>' in prompt
    assert '<excerpt page="17-18" section="Signing up">' in prompt
    assert prompt.endswith("Question: What does Part B cover?")


def test_answer_reports_cited_pages_and_sources() -> None:
    llm = FakeLLM("Outpatient care [p. 32]. Sign up at 65 [pp. 17-18].")
    answer = answer_question("q", FakeStore([PART_B, ENROLLMENT]), llm)

    assert answer.cited_pages == [17, 18, 32]
    assert answer.unsupported_pages == []
    assert [source.chunk for source in answer.sources] == [PART_B, ENROLLMENT]


def test_cited_sources_are_only_the_excerpts_the_answer_cites() -> None:
    answer = answer_question("q", FakeStore([PART_B, ENROLLMENT]), FakeLLM("Outpatient care [p. 32]."))
    assert [source.chunk for source in answer.cited_sources] == [PART_B]


def test_citing_one_page_of_a_multi_page_excerpt_counts_as_citing_it() -> None:
    answer = answer_question("q", FakeStore([PART_B, ENROLLMENT]), FakeLLM("At 65 [p. 18]."))
    assert [source.chunk for source in answer.cited_sources] == [ENROLLMENT]


def test_citation_of_a_page_not_in_the_sources_is_flagged() -> None:
    llm = FakeLLM("Outpatient care [p. 32], and dental care [p. 58].")
    answer = answer_question("q", FakeStore([PART_B]), llm)

    assert answer.unsupported_pages == [58]


def test_answer_without_citations_is_flagged() -> None:
    answer = answer_question("q", FakeStore([PART_B]), FakeLLM("Part B covers outpatient care."))
    assert answer.is_uncited


def test_not_found_reply_is_not_flagged_as_uncited() -> None:
    answer = answer_question("q", FakeStore([PART_B]), FakeLLM(NOT_FOUND))
    assert not answer.is_uncited


def test_no_retrieved_excerpts_means_no_model_call() -> None:
    llm = FakeLLM("should not be used")
    answer = answer_question("q", FakeStore([]), llm)

    assert answer.text == NOT_FOUND
    assert llm.calls == []


@pytest.mark.parametrize(
    ("text", "pages"),
    [
        ("Covered [p. 9].", [9]),
        ("Covered [p.9] and [p. 12].", [9, 12]),
        ("Spans [pp. 25-26].", [25, 26]),
        ("En dash [pp. 25–27].", [25, 26, 27]),
        ("Listed [p. 9, p. 12].", [9, 12]),
        ("Repeated [p. 9] [p. 9].", [9]),
        ("No citation, and a bracket [note] or year 2027.", []),
    ],
)
def test_cited_pages_parses_citation_formats(text: str, pages: list[int]) -> None:
    assert cited_pages(text) == pages


def test_format_context_escapes_quotes_in_section_titles() -> None:
    chunk = make_chunk("c3", 'The "Welcome" visit', "Body.", 57, 57)
    context = format_context([SearchResult(chunk, score=1.0)])
    assert 'section="The \'Welcome\' visit"' in context
