"""BM25-based keyword index for lexical retrieval."""

import re
import math
import logging
from typing import List, Dict, Optional
from collections import Counter

from adaptive_intelligence.indexes.base import BaseIndex, RetrievalResult
from adaptive_intelligence.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class KeywordIndex(BaseIndex):
    """BM25-based keyword index for exact term matching."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: List[Chunk] = []
        self._tokenized_docs: List[List[str]] = []
        self._doc_freqs: Dict[str, int] = {}
        self._avg_dl: float = 0.0
        self._idf_cache: Dict[str, float] = {}
        self._built = False

    @property
    def index_type(self) -> str:
        return "keyword"

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        # Remove very short tokens
        return [t for t in tokens if len(t) > 1]

    def add(self, chunks: List[Chunk]) -> int:
        """Add chunks to the keyword index."""
        if not chunks:
            return 0

        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            self._chunks.append(chunk)
            self._tokenized_docs.append(tokens)

        self._build_index()
        logger.info(f"Added {len(chunks)} chunks to keyword index")
        return len(chunks)

    def _build_index(self):
        """Build BM25 statistics."""
        self._doc_freqs.clear()
        self._idf_cache.clear()

        n_docs = len(self._tokenized_docs)
        if n_docs == 0:
            return

        # Document frequencies
        for doc_tokens in self._tokenized_docs:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

        # Average document length
        total_tokens = sum(len(doc) for doc in self._tokenized_docs)
        self._avg_dl = total_tokens / n_docs if n_docs > 0 else 0

        # Pre-compute IDF
        for term, df in self._doc_freqs.items():
            self._idf_cache[term] = math.log(
                (n_docs - df + 0.5) / (df + 0.5) + 1.0
            )

        self._built = True

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """BM25 search."""
        if not self._built or not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i, doc_tokens in enumerate(self._tokenized_docs):
            score = self._score_document(query_tokens, doc_tokens)
            scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                # Normalize score to [0, 1] range approximately
                normalized = min(1.0, score / (score + 5.0))
                results.append(RetrievalResult(
                    chunk=self._chunks[idx],
                    score=normalized,
                    index_type="keyword",
                ))

        return results

    def _score_document(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Calculate BM25 score for a document."""
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0

        tf_map = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            if term not in self._idf_cache:
                continue

            tf = tf_map.get(term, 0)
            idf = self._idf_cache[term]

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_dl)
            score += idf * (numerator / denominator)

        return score

    def clear(self) -> None:
        """Clear the keyword index."""
        self._chunks.clear()
        self._tokenized_docs.clear()
        self._doc_freqs.clear()
        self._idf_cache.clear()
        self._avg_dl = 0.0
        self._built = False

    def count(self) -> int:
        return len(self._chunks)
