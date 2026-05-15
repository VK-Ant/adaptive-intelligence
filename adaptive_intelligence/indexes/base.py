"""Base interface for all retrieval indexes."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from adaptive_intelligence.ingestion.chunker import Chunk


@dataclass
class RetrievalResult:
    """A single retrieval result with score."""
    chunk: Chunk
    score: float
    index_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"RetrievalResult(score={self.score:.3f}, index='{self.index_type}', doc='{self.chunk.source_document}')"


class BaseIndex(ABC):
    """Abstract base class for all retrieval indexes."""

    @abstractmethod
    def add(self, chunks: List[Chunk]) -> int:
        """Add chunks to the index. Returns count added."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search the index. Returns ranked results."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the index."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return number of items in the index."""
        pass

    @property
    @abstractmethod
    def index_type(self) -> str:
        """Return the type of this index."""
        pass
