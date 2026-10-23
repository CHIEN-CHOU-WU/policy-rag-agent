"""Ask a question from the command line.

    uv run python -m policy_rag.ask "What does Part B cover?"                # RAG: search once, answer
    uv run python -m policy_rag.ask --agent "How do I add dental coverage?"  # agent: research, then answer

The model provider comes from ``LLM_PROVIDER`` (``.env``; Ollama by default),
and can be overridden per run: ``LLM_PROVIDER=anthropic uv run python -m policy_rag.ask "..."``.
"""

import argparse
import sys
from collections.abc import Iterable

from policy_rag.agent import AgentAnswer, describe_step, run_agent
from policy_rag.config import load_settings
from policy_rag.llm.base import LLMError
from policy_rag.llm.factory import create_llm_client
from policy_rag.rag import Answer, answer_question
from policy_rag.store.factory import create_page_store, create_vector_store
from policy_rag.tools import PolicyTools


def main() -> int:
    parser = argparse.ArgumentParser(description="Answer a question about the policy documents.")
    parser.add_argument("question")
    parser.add_argument("--agent", action="store_true", help="let the model research with tools")
    parser.add_argument("-k", type=int, default=5, help="excerpts to retrieve (RAG mode)")
    args = parser.parse_args()

    settings = load_settings()
    store = create_vector_store(settings)
    if store.count() == 0:
        print("The index is empty. Run `uv run python -m policy_rag.ingest` first.", file=sys.stderr)
        return 1

    llm = create_llm_client(settings)
    try:
        if args.agent:
            tools = PolicyTools(store, create_page_store(settings))
            _print_agent_answer(run_agent(args.question, llm, tools))
        else:
            _print_answer(answer_question(args.question, store, llm, k=args.k))
    except LLMError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


def _print_answer(answer: Answer) -> None:
    print(answer.text)
    # Only excerpts the answer cites are its sources; listing everything that
    # was retrieved would suggest support that doesn't exist.
    _print_sources(
        f"p. {source.chunk.page_label:<7} {source.chunk.section_title}"
        for source in answer.cited_sources
    )
    _print_warnings(answer.is_uncited, answer.unsupported_pages)


def _print_agent_answer(answer: AgentAnswer) -> None:
    print("Research steps:")
    for number, step in enumerate(answer.steps, start=1):
        print(f"  {number}. {describe_step(step)}")
    print()
    print(answer.text)
    _print_sources(f"p. {page:<7} {label}" for page, label in answer.cited_sources)
    _print_warnings(answer.is_uncited, answer.unsupported_pages)


def _print_sources(lines: Iterable[str]) -> None:
    unique = list(dict.fromkeys(lines))
    if unique:
        print("\nSources:")
        print("\n".join(f"  {line}" for line in unique))


def _print_warnings(is_uncited: bool, unsupported_pages: list[int]) -> None:
    if is_uncited:
        print("\nWarning: the answer cites no pages, so it cannot be verified against the sources.")
    if unsupported_pages:
        pages = ", ".join(str(page) for page in unsupported_pages)
        print(f"\nWarning: the answer cites page(s) {pages}, which the sources never showed.")


if __name__ == "__main__":
    sys.exit(main())
