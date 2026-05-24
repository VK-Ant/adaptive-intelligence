"""Cross-encoder reranking — v3 addition.

Re-scores top-N retrieved chunks using a cross-encoder model.
Falls back gracefully if sentence-transformers not installed.
"""

import logging
from typing import List, Optional, Tuple

from adaptive_intelligence.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for retrieved chunks.

    Uses a cross-encoder model to re-score query-chunk pairs.
    Much more accurate than bi-encoder similarity but slower.
    Applied only to top-N candidates (default 20).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 top_n: int = 20, device: str = "cpu"):
        self.model_name = model_name
        self.top_n = top_n
        self.device = device
        self._model = None
        self._available = False
        self._init_model()

    def _init_model(self):
        """Try to load cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            self._available = True
            logger.info(f"Cross-encoder loaded: {self.model_name}")
        except ImportError:
            logger.info("Cross-encoder not available (install sentence-transformers)")
            self._available = False
        except Exception as e:
            logger.warning(f"Cross-encoder init failed: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def rerank(self, query: str, chunks: List[Chunk],
               top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Rerank chunks by cross-encoder relevance score.

        Args:
            query: The search query
            chunks: Retrieved chunks to rerank
            top_k: Number of top results to return

        Returns:
            List of (chunk, score) tuples, sorted by score descending
        """
        if not self._available or not chunks:
            return [(c, 1.0 - i * 0.1) for i, c in enumerate(chunks[:top_k])]

        # Limit to top_n candidates
        candidates = chunks[:self.top_n]

        # Create query-chunk pairs
        pairs = [(query, c.content[:512]) for c in candidates]

        try:
            scores = self._model.predict(pairs)
            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return [(c, 1.0 - i * 0.1) for i, c in enumerate(chunks[:top_k])]


class KeywordReranker:
    """Simple keyword-based reranker (no model dependency).

    Scores chunks by query term overlap. Used as fallback when
    cross-encoder is not available.
    """

    def rerank(self, query: str, chunks: List[Chunk],
               top_k: int = 5) -> List[Tuple[Chunk, float]]:
        query_terms = set(query.lower().split())
        scored = []
        for chunk in chunks:
            chunk_terms = set(chunk.content.lower().split())
            overlap = len(query_terms & chunk_terms)
            score = overlap / max(len(query_terms), 1)
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
