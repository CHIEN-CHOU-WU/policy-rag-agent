"""Measure retrieval quality: does the right page come back, and how high?

Runs every question in ``retrieval_questions.yaml`` against the vector store
(no LLM, so it is free and fast) and reports:

- hit@k: share of questions with a correct page among the top k results
- MRR:   mean reciprocal rank of the first correct result (1.0 = always first)

Usage: ``uv run python eval/run_retrieval_eval.py``
"""

import sys
from pathlib import Path

import yaml

from policy_rag.config import load_settings
from policy_rag.models import Chunk
from policy_rag.store.factory import create_vector_store

QUESTIONS_FILE = Path(__file__).with_name("retrieval_questions.yaml")
CUTOFFS = (1, 3, 5)
TOP_K = max(CUTOFFS)


def first_hit_rank(chunks: list[Chunk], expected_pages: set[int]) -> int | None:
    """1-based rank of the first chunk whose page range includes an expected page."""
    for rank, chunk in enumerate(chunks, start=1):
        if expected_pages & set(range(chunk.page_start, chunk.page_end + 1)):
            return rank
    return None


def main() -> int:
    questions = yaml.safe_load(QUESTIONS_FILE.read_text())["questions"]
    store = create_vector_store(load_settings())
    if store.count() == 0:
        print("The index is empty. Run `uv run python -m policy_rag.ingest` first.", file=sys.stderr)
        return 1

    ranks: list[int | None] = []
    print(f"{'rank':>4}  {'expected':<10} {'retrieved pages':<28} question")
    for item in questions:
        chunks = [result.chunk for result in store.search(item["question"], k=TOP_K)]
        rank = first_hit_rank(chunks, set(item["pages"]))
        ranks.append(rank)
        expected = ",".join(str(page) for page in item["pages"])
        retrieved = " ".join(chunk.page_label for chunk in chunks)
        print(f"{rank or '-':>4}  {expected:<10} {retrieved:<28} {item['question']}")

    print()
    for k in CUTOFFS:
        hits = sum(1 for rank in ranks if rank is not None and rank <= k)
        print(f"hit@{k}  {hits / len(ranks):.2f}  ({hits}/{len(ranks)})")
    mrr = sum(1 / rank for rank in ranks if rank is not None) / len(ranks)
    print(f"MRR    {mrr:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
