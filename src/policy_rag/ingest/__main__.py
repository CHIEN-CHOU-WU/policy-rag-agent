"""Build the search index: ``uv run python -m policy_rag.ingest``.

Parses every PDF in the raw data directory, chunks it by section, and rebuilds
the vector store from scratch so no stale chunks survive a re-run.
"""

import sys
from collections import Counter

from policy_rag.config import load_settings
from policy_rag.ingest.chunk import chunk_document
from policy_rag.ingest.parse import parse_pdf
from policy_rag.models import Chunk
from policy_rag.store.factory import create_vector_store


def main() -> int:
    settings = load_settings()
    pdf_paths = sorted(settings.raw_data_dir.glob("*.pdf"))
    if not pdf_paths:
        print(
            f"No PDFs found in {settings.raw_data_dir}/. "
            "Run `uv run python scripts/download_data.py` first.",
            file=sys.stderr,
        )
        return 1

    total_pages = 0
    numbering: Counter[str] = Counter()
    chunks: list[Chunk] = []
    for path in pdf_paths:
        print(f"Parsing {path.name} ...")
        document = parse_pdf(path)
        total_pages += len(document.pages)
        numbering.update(page.page_numbering for page in document.pages)
        chunks.extend(chunk_document(document))

    print(f"Indexing {len(chunks)} chunks into {settings.chroma_dir}/ ...")
    store = create_vector_store(settings)
    store.reset()
    store.add(chunks)

    _print_stats(len(pdf_paths), total_pages, numbering, chunks, store.count())
    return 0


def _print_stats(
    documents: int, pages: int, numbering: Counter[str], chunks: list[Chunk], stored: int
) -> None:
    tables = sum(chunk.content_type == "table" for chunk in chunks)
    average = sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0
    numbering_summary = ", ".join(f"{source}: {n}" for source, n in numbering.most_common())

    print()
    print("Ingest complete")
    print(f"  Documents        {documents}")
    print(f"  Pages            {pages}  ({numbering_summary})")
    print(f"  Chunks           {len(chunks)}  (stored: {stored})")
    print(f"  Avg chunk length {average:.0f} chars")
    print(f"  Tables           {tables}")


if __name__ == "__main__":
    sys.exit(main())
