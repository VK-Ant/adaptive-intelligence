"""Ingestion engine — v2 with incremental, parallel, SQL, progress.

Orchestrates: parse → chunk → index with full lifecycle management.
"""

import logging
import time
import signal
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from adaptive_intelligence.core.config import AdaptiveConfig
from adaptive_intelligence.ingestion.parser import DocumentParser, ParsedDocument, SQLConnector
from adaptive_intelligence.ingestion.chunker import DocumentChunker, Chunk

logger = logging.getLogger(__name__)


class IngestionStats:
    """Statistics from an ingestion run."""

    def __init__(self):
        self.total_files = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.total_chunks = 0
        self.total_tables = 0
        self.rejected_chunks = 0
        self.deduplicated_chunks = 0
        self.file_types: Dict[str, int] = {}
        self.errors: List[str] = []
        self.duration_seconds: float = 0.0

    def display(self) -> str:
        lines = [f"Documents: {self.successful}/{self.total_files}"]
        for ft, count in sorted(self.file_types.items()):
            lines.append(f"  {ft.upper()}: {count}")
        lines.extend([
            f"Chunks created: {self.total_chunks}",
            f"Tables extracted: {self.total_tables}",
        ])
        if self.rejected_chunks > 0:
            lines.append(f"Chunks rejected (quality): {self.rejected_chunks}")
        if self.deduplicated_chunks > 0:
            lines.append(f"Chunks deduplicated: {self.deduplicated_chunks}")
        if self.failed > 0:
            lines.append(f"Failed: {self.failed}")
        if self.skipped > 0:
            lines.append(f"Skipped: {self.skipped}")
        lines.append(f"Duration: {self.duration_seconds:.1f}s")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "\n".join(lines)


class IngestionEngine:
    """Orchestrates document ingestion — v2 with incremental + parallel."""

    def __init__(self, config: Optional[AdaptiveConfig] = None,
                 page_mode: bool = False,
                 min_chunk_quality: float = 0.3,
                 deduplicate: bool = False):
        self.config = config or AdaptiveConfig()
        self.page_mode = page_mode
        self.parser = DocumentParser()
        self.sql_connector = SQLConnector()
        self.chunker = DocumentChunker(
            config=self.config.chunking,
            page_mode=page_mode,
            min_quality=min_chunk_quality,
            deduplicate=deduplicate,
        )
        self.documents: Dict[str, ParsedDocument] = {}  # filepath -> doc
        self.chunks: List[Chunk] = []
        self._doc_chunks: Dict[str, List[str]] = {}  # filepath -> [chunk_ids]

    def ingest(self, source: Union[str, List[str]],
               parallel: bool = False,
               workers: int = 4,
               on_progress: Optional[Callable] = None,
               tables: Optional[List[str]] = None,
               query: Optional[str] = None) -> IngestionStats:
        """Ingest documents from files, directories, or SQL.

        Args:
            source: File path, directory, list of paths, or SQL connection string
            parallel: Enable parallel processing
            workers: Number of parallel workers
            on_progress: Callback(done, total) for progress tracking
            tables: SQL tables to ingest (for SQL sources)
            query: Custom SQL query (for SQL sources)
        """
        stats = IngestionStats()
        start_time = time.time()

        sources = source if isinstance(source, list) else [source]

        for src in sources:
            # Detect SQL connection strings
            if self._is_sql_source(src):
                self._ingest_sql(src, stats, tables=tables, query=query)
            elif Path(src).is_file():
                self._ingest_file(str(src), stats)
            elif Path(src).is_dir():
                if parallel:
                    self._ingest_directory_parallel(str(src), stats, workers, on_progress)
                else:
                    self._ingest_directory(str(src), stats, on_progress)
            else:
                stats.errors.append(f"Source not found: {src}")
                logger.warning(f"Source not found: {src}")

        stats.duration_seconds = time.time() - start_time

        logger.info(
            f"Ingestion complete: {stats.successful} docs, "
            f"{stats.total_chunks} chunks in {stats.duration_seconds:.1f}s"
        )
        return stats

    def ingest_chunks(self) -> List[Chunk]:
        """Return all chunks ready for indexing (called by engine)."""
        return self.chunks

    def remove(self, filename: str) -> int:
        """Remove a document and its chunks. Returns chunks removed."""
        # Find matching document
        to_remove = None
        for fp, doc in self.documents.items():
            if doc.filename == filename or fp == filename:
                to_remove = fp
                break

        if not to_remove:
            logger.warning(f"Document not found: {filename}")
            return 0

        # Get chunk IDs to remove
        chunk_ids = set(self._doc_chunks.get(to_remove, []))
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.chunk_id not in chunk_ids]
        removed = before - len(self.chunks)

        del self.documents[to_remove]
        if to_remove in self._doc_chunks:
            del self._doc_chunks[to_remove]

        logger.info(f"Removed {filename}: {removed} chunks")
        return removed

    def update(self, filepath: str) -> IngestionStats:
        """Re-ingest a single document (remove old + add new)."""
        path = Path(filepath)
        self.remove(path.name)
        stats = IngestionStats()
        self._ingest_file(str(path), stats)
        return stats

    def _ingest_file(self, filepath: str, stats: IngestionStats) -> Optional[ParsedDocument]:
        stats.total_files += 1
        try:
            doc = self.parser.parse(filepath)

            # Skip if parse produced only errors
            if doc.content.startswith("[") and doc.parse_errors:
                stats.failed += 1
                stats.errors.extend(doc.parse_errors)
                return None

            chunks = self.chunker.chunk_document(doc)

            self.documents[filepath] = doc
            self._doc_chunks[filepath] = [c.chunk_id for c in chunks]
            self.chunks.extend(chunks)

            stats.successful += 1
            stats.total_chunks += len(chunks)
            stats.total_tables += len(doc.tables)
            ft = doc.file_type
            stats.file_types[ft] = stats.file_types.get(ft, 0) + 1

            if doc.parse_errors:
                stats.errors.extend(doc.parse_errors)

            return doc

        except Exception as e:
            stats.failed += 1
            stats.errors.append(f"{filepath}: {e}")
            logger.warning(f"Failed to ingest {filepath}: {e}")
            return None

    def _ingest_directory(self, directory: str, stats: IngestionStats,
                          on_progress: Optional[Callable] = None):
        files = self._list_supported_files(directory)
        total = len(files)

        for i, file_path in enumerate(files):
            self._ingest_file(str(file_path), stats)
            if on_progress:
                on_progress(i + 1, total)

    def _ingest_directory_parallel(self, directory: str, stats: IngestionStats,
                                   workers: int, on_progress: Optional[Callable] = None):
        files = self._list_supported_files(directory)
        total = len(files)
        done = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.parser.parse, str(fp)): str(fp)
                for fp in files
            }

            for future in as_completed(futures):
                filepath = futures[future]
                stats.total_files += 1
                try:
                    doc = future.result()
                    if doc.content.startswith("[") and doc.parse_errors:
                        stats.failed += 1
                        continue

                    chunks = self.chunker.chunk_document(doc)
                    self.documents[filepath] = doc
                    self._doc_chunks[filepath] = [c.chunk_id for c in chunks]
                    self.chunks.extend(chunks)

                    stats.successful += 1
                    stats.total_chunks += len(chunks)
                    stats.total_tables += len(doc.tables)
                    ft = doc.file_type
                    stats.file_types[ft] = stats.file_types.get(ft, 0) + 1

                except Exception as e:
                    stats.failed += 1
                    stats.errors.append(f"{filepath}: {e}")

                done += 1
                if on_progress:
                    on_progress(done, total)

    def _ingest_sql(self, connection_string: str, stats: IngestionStats,
                    tables: Optional[List[str]] = None,
                    query: Optional[str] = None):
        try:
            docs = self.sql_connector.parse(
                connection_string, tables=tables, query=query
            )
            for doc in docs:
                chunks = self.chunker.chunk_document(doc)
                key = f"sql:{doc.filename}"
                self.documents[key] = doc
                self._doc_chunks[key] = [c.chunk_id for c in chunks]
                self.chunks.extend(chunks)

                stats.total_files += 1
                stats.successful += 1
                stats.total_chunks += len(chunks)
                stats.total_tables += len(doc.tables)
                stats.file_types["sql"] = stats.file_types.get("sql", 0) + 1

        except Exception as e:
            stats.failed += 1
            stats.errors.append(f"SQL ingest failed: {e}")
            logger.error(f"SQL ingest failed: {e}")

    def _is_sql_source(self, source: str) -> bool:
        sql_prefixes = ("sqlite:///", "postgresql://", "postgres://",
                        "mysql://", "mssql://", "oracle://")
        return any(source.startswith(p) for p in sql_prefixes)

    def _list_supported_files(self, directory: str) -> List[Path]:
        path = Path(directory)
        files = []
        for fp in sorted(path.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in DocumentParser.SUPPORTED_EXTENSIONS:
                files.append(fp)
        return files

    def get_chunks(self) -> List[Chunk]:
        return self.chunks

    def get_documents(self) -> List[ParsedDocument]:
        return list(self.documents.values())

    def get_document_list(self) -> List[Dict[str, Any]]:
        """List all ingested documents with metadata."""
        return [
            {
                "filename": doc.filename,
                "file_type": doc.file_type,
                "pages": doc.page_count,
                "chunks": len(self._doc_chunks.get(fp, [])),
                "tables": len(doc.tables),
                "chars": doc.char_count,
                "errors": len(doc.parse_errors),
            }
            for fp, doc in self.documents.items()
        ]

    def clear(self):
        self.documents.clear()
        self.chunks.clear()
        self._doc_chunks.clear()
