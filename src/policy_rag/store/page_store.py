"""Storage for full page text, used to read a whole page in context.

A port (``PageStore``) and a JSON-file adapter. Pages are small and few, so a
single JSON file is enough and keeps the index inspectable by hand.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from policy_rag.models import PageText


class PageStore(ABC):
    @abstractmethod
    def save(self, pages: Sequence[PageText]) -> None:
        """Replace all stored pages."""

    @abstractmethod
    def get(self, page_number: int) -> list[PageText]:
        """Pages with this citation page number (one per source document)."""


class JsonPageStore(PageStore):
    def __init__(self, path: Path) -> None:
        self._path = path

    def save(self, pages: Sequence[PageText]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        records = [asdict(page) for page in pages]
        self._path.write_text(json.dumps(records, ensure_ascii=False, indent=1))

    def get(self, page_number: int) -> list[PageText]:
        return [page for page in self._load() if page.page_number == page_number]

    def _load(self) -> list[PageText]:
        if not self._path.exists():
            return []
        return [PageText(**record) for record in json.loads(self._path.read_text())]
