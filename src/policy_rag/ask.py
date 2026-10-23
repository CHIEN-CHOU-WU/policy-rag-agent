"""Ask a question from the command line: ``uv run python -m policy_rag.ask "..."``.

The model provider comes from ``LLM_PROVIDER`` (``.env``), and can be
overridden per run: ``LLM_PROVIDER=ollama uv run python -m policy_rag.ask "..."``.
"""

import argparse
import sys

from policy_rag.config import load_settings
from policy_rag.llm.base import LLMError
from policy_rag.llm.factory import create_llm_client
from policy_rag.rag import Answer, answer_question
from policy_rag.store.factory import create_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Answer a question about the policy documents.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="number of excerpts to retrieve")
    args = parser.parse_args()

    settings = load_settings()
    store = create_vector_store(settings)
    if store.count() == 0:
        print("The index is empty. Run `uv run python -m policy_rag.ingest` first.", file=sys.stderr)
        return 1

    try:
        answer = answer_question(args.question, store, create_llm_client(settings), k=args.k)
    except LLMError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    _print_answer(answer)
    return 0


def _print_answer(answer: Answer) -> None:
    print(answer.text)
    # Only excerpts the answer cites are its sources; listing everything that
    # was retrieved would suggest support that doesn't exist.
    lines = dict.fromkeys(
        f"  p. {source.chunk.page_label:<7} {source.chunk.section_title}"
        for source in answer.cited_sources
    )
    if lines:
        print("\nSources:")
        print("\n".join(lines))
    if answer.is_uncited:
        print("\nWarning: the answer cites no pages, so it cannot be verified against the sources.")
    if answer.unsupported_pages:
        pages = ", ".join(str(page) for page in answer.unsupported_pages)
        print(f"\nWarning: the answer cites page(s) {pages}, which are not in the sources above.")


if __name__ == "__main__":
    sys.exit(main())
