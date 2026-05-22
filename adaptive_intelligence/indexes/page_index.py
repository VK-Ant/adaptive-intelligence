"""Page-level BM25 index for vectorless mode.

No embeddings. No vector DB. Pure Python keyword search over full pages.
Provides page-number citations instead of chunk IDs.
"""

import logging
from typing import List, Optional

from adaptive_intelligence.indexes.base import BaseIndex, RetrievalResult
from adaptive_intelligence.indexes.keyword_index import KeywordIndex
from adaptive_intelligence.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class PageIndex(BaseIndex):
    """Page-level BM25 index. Zero external dependencies."""

    def __init__(self):
        self._bm25 = KeywordIndex()
        self._page_chunks: List[Chunk] = []

    @property
    def index_type(self) -> str:
        return "page"

    def add(self, chunks: List[Chunk]) -> int:
        """Add page-level chunks to the index."""
        page_chunks = [c for c in chunks if c.is_page]
        if not page_chunks:
            # If no page chunks, accept all chunks
            page_chunks = chunks

        self._page_chunks.extend(page_chunks)
        return self._bm25.add(page_chunks)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search pages by keyword matching."""
        results = self._bm25.search(query, top_k=top_k)
        # Ensure page metadata is present
        for r in results:
            r.index_type = "page"
        return results

    def clear(self) -> None:
        self._bm25.clear()
        self._page_chunks.clear()

    def count(self) -> int:
        return self._bm25.count()

    def save(self, filepath: str):
        """Persist page index to disk."""
        self._bm25.save(filepath)

    def load(self, filepath: str):
        """Load page index from disk."""
        self._bm25.load(filepath)
