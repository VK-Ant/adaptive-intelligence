"""Document chunking with smart boundary detection."""

import hashlib
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from adaptive_intelligence.core.config import ChunkingConfig
from adaptive_intelligence.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of text from a document."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_document: str = ""
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    section: str = ""
    is_table: bool = False
    entities: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(
                f"{self.doc_id}:{self.chunk_index}:{self.content[:100]}".encode()
            ).hexdigest()[:12]

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Chunk(id='{self.chunk_id}', doc='{self.source_document}', len={len(self.content)}, preview='{preview}...')"


class DocumentChunker:
    """Chunk documents with smart boundary detection and overlap."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self._sentence_endings = re.compile(r"(?<=[.!?])\s+")
        self._paragraph_breaks = re.compile(r"\n\s*\n")

    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """Chunk a parsed document into retrievable pieces."""
        chunks = []

        # Chunk main text content
        text_chunks = self._chunk_text(
            document.content,
            doc_id=document.doc_id,
            source=document.filename,
        )
        chunks.extend(text_chunks)

        # Chunk tables separately (preserve structure)
        for i, table in enumerate(document.tables):
            table_chunk = self._chunk_table(table, document.doc_id, document.filename, i)
            chunks.extend(table_chunk)

        # Set chunk indices
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        logger.info(f"Created {len(chunks)} chunks from {document.filename}")
        return chunks

    def _chunk_text(
        self,
        text: str,
        doc_id: str = "",
        source: str = "",
    ) -> List[Chunk]:
        """Chunk text with overlap and boundary respect."""
        if not text.strip():
            return []

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        min_size = self.config.min_chunk_size

        # Split into paragraphs first
        paragraphs = self._paragraph_breaks.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_text = ""
        current_start = 0
        char_offset = 0

        for para in paragraphs:
            if len(current_text) + len(para) + 1 <= chunk_size:
                current_text += ("\n\n" if current_text else "") + para
            else:
                # Current chunk is full
                if len(current_text) >= min_size:
                    chunks.append(Chunk(
                        chunk_id="",
                        doc_id=doc_id,
                        content=current_text.strip(),
                        source_document=source,
                        start_char=current_start,
                        end_char=current_start + len(current_text),
                    ))

                # Handle overlap: carry last overlap chars
                if overlap > 0 and current_text:
                    overlap_text = current_text[-overlap:]
                    # Try to start overlap at sentence boundary
                    sentences = self._sentence_endings.split(overlap_text)
                    if len(sentences) > 1:
                        overlap_text = sentences[-1]
                    current_text = overlap_text + "\n\n" + para
                else:
                    current_text = para
                current_start = char_offset

            char_offset += len(para) + 2  # +2 for \n\n

        # Final chunk
        if len(current_text) >= min_size:
            chunks.append(Chunk(
                chunk_id="",
                doc_id=doc_id,
                content=current_text.strip(),
                source_document=source,
                start_char=current_start,
                end_char=current_start + len(current_text),
            ))
        elif current_text.strip() and chunks:
            # Append small remainder to last chunk
            chunks[-1].content += "\n\n" + current_text.strip()
            chunks[-1].end_char += len(current_text) + 2

        return chunks

    def _chunk_table(
        self,
        table: Dict[str, Any],
        doc_id: str,
        source: str,
        table_index: int,
    ) -> List[Chunk]:
        """Chunk a table, keeping structure intact."""
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        if not rows:
            return []

        # Build table as text
        header_line = " | ".join(str(h) for h in headers) if headers else ""
        row_lines = []
        for row in rows:
            row_lines.append(" | ".join(str(cell) for cell in row))

        # If table fits in one chunk, keep it whole
        full_table = header_line + "\n" + "\n".join(row_lines) if header_line else "\n".join(row_lines)

        if len(full_table) <= self.config.chunk_size:
            return [Chunk(
                chunk_id="",
                doc_id=doc_id,
                content=full_table,
                source_document=source,
                is_table=True,
                metadata={
                    "table_index": table_index,
                    "row_count": len(rows),
                    "columns": headers,
                    "page": table.get("page"),
                },
            )]

        # Split large tables: each chunk gets headers + subset of rows
        chunks = []
        batch_size = max(1, self.config.chunk_size // (len(header_line) + 50))
        for i in range(0, len(row_lines), batch_size):
            batch = row_lines[i : i + batch_size]
            chunk_content = header_line + "\n" + "\n".join(batch) if header_line else "\n".join(batch)
            chunks.append(Chunk(
                chunk_id="",
                doc_id=doc_id,
                content=chunk_content,
                source_document=source,
                is_table=True,
                metadata={
                    "table_index": table_index,
                    "row_range": f"{i}-{i + len(batch)}",
                    "columns": headers,
                    "page": table.get("page"),
                },
            ))

        return chunks
