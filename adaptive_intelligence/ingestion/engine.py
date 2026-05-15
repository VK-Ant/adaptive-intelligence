"""Ingestion engine - orchestrates document parsing, chunking, and indexing."""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from adaptive_intelligence.core.config import AdaptiveConfig
from adaptive_intelligence.ingestion.parser import DocumentParser, ParsedDocument
from adaptive_intelligence.ingestion.chunker import DocumentChunker, Chunk

logger = logging.getLogger(__name__)


class IngestionStats:
    """Statistics from an ingestion run."""

    def __init__(self):
        self.total_files = 0
        self.successful = 0
        self.failed = 0
        self.total_chunks = 0
        self.total_tables = 0
        self.total_entities = 0
        self.file_types: Dict[str, int] = {}
        self.duration_seconds: float = 0.0

    def display(self) -> str:
        lines = [
            f"Documents: {self.total_files}",
        ]
        for ft, count in sorted(self.file_types.items()):
            lines.append(f"  • {ft.upper()}: {count}")
        lines.extend([
            f"Chunks created: {self.total_chunks}",
            f"Tables extracted: {self.total_tables}",
            f"Failed: {self.failed}",
            f"Duration: {self.duration_seconds:.1f}s",
        ])
        return "\n".join(lines)


class IngestionEngine:
    """Orchestrates document ingestion pipeline."""

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.parser = DocumentParser()
        self.chunker = DocumentChunker(self.config.chunking)
        self.documents: List[ParsedDocument] = []
        self.chunks: List[Chunk] = []

    def ingest(self, source: str) -> IngestionStats:
        """Ingest documents from a file or directory."""
        stats = IngestionStats()
        start_time = time.time()

        path = Path(source)
        if path.is_file():
            docs = [self._ingest_file(str(path), stats)]
            docs = [d for d in docs if d is not None]
        elif path.is_dir():
            docs = self._ingest_directory(str(path), stats)
        else:
            raise FileNotFoundError(f"Source not found: {source}")

        # Chunk all documents
        all_chunks = []
        for doc in docs:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            stats.total_tables += len(doc.tables)

        self.documents.extend(docs)
        self.chunks.extend(all_chunks)
        stats.total_chunks = len(all_chunks)
        stats.duration_seconds = time.time() - start_time

        logger.info(
            f"Ingestion complete: {stats.successful} docs, "
            f"{stats.total_chunks} chunks in {stats.duration_seconds:.1f}s"
        )
        return stats

    def _ingest_file(self, filepath: str, stats: IngestionStats) -> Optional[ParsedDocument]:
        """Ingest a single file."""
        stats.total_files += 1
        try:
            doc = self.parser.parse(filepath)
            stats.successful += 1
            ft = doc.file_type
            stats.file_types[ft] = stats.file_types.get(ft, 0) + 1
            return doc
        except Exception as e:
            stats.failed += 1
            logger.warning(f"Failed to ingest {filepath}: {e}")
            return None

    def _ingest_directory(self, directory: str, stats: IngestionStats) -> List[ParsedDocument]:
        """Ingest all supported files from a directory."""
        docs = []
        parsed = self.parser.parse_directory(directory)
        for doc in parsed:
            stats.total_files += 1
            stats.successful += 1
            ft = doc.file_type
            stats.file_types[ft] = stats.file_types.get(ft, 0) + 1
            docs.append(doc)
        return docs

    def get_chunks(self) -> List[Chunk]:
        """Return all chunks from ingested documents."""
        return self.chunks

    def get_documents(self) -> List[ParsedDocument]:
        """Return all parsed documents."""
        return self.documents

    def clear(self):
        """Clear all ingested data."""
        self.documents.clear()
        self.chunks.clear()
