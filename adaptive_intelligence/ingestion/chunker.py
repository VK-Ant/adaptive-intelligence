"""Document chunking — v2 with page index, quality scoring, dedup.

Supports three modes:
  - Chunk mode (v1): paragraph-boundary chunks with overlap
  - Page mode (v2): full page as retrieval unit
  - Hybrid mode (v2): page-level coarse + chunk-level fine
"""

import hashlib
import logging
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import Counter

from adaptive_intelligence.core.config import ChunkingConfig
from adaptive_intelligence.ingestion.parser import ParsedDocument, PageContent

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
    is_page: bool = False
    page_number: int = 0
    entities: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    prev_chunk_id: str = ""
    next_chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(
                f"{self.doc_id}:{self.chunk_index}:{self.content[:100]}".encode()
            ).hexdigest()[:12]

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        tag = "[PAGE]" if self.is_page else "[TABLE]" if self.is_table else ""
        return f"Chunk(id='{self.chunk_id}', {tag} doc='{self.source_document}', len={len(self.content)})"

    def __eq__(self, other):
        if isinstance(other, Chunk):
            return self.chunk_id == other.chunk_id
        return False

    def __hash__(self):
        return hash(self.chunk_id)


class ChunkQualityScorer:
    """Score chunk quality to reject garbage before indexing."""

    def score(self, chunk: Chunk) -> float:
        text = chunk.content
        score = 1.0

        # Too short
        word_count = len(text.split())
        if word_count < 5:
            score -= 0.5
        elif word_count < 10:
            score -= 0.2

        # Too much whitespace (garbled OCR)
        if len(text) > 0:
            whitespace_ratio = text.count(" ") / len(text)
            if whitespace_ratio > 0.6:
                score -= 0.3

        # Repetitive content
        words = text.lower().split()
        if words:
            counter = Counter(words)
            most_common_count = counter.most_common(1)[0][1]
            if most_common_count / len(words) > 0.4:
                score -= 0.3

        # No alphabetic characters
        if len(text) > 0:
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.3:
                score -= 0.2

        # Mostly numbers/symbols (not a table)
        if not chunk.is_table and len(text) > 0:
            digit_ratio = sum(c.isdigit() for c in text) / len(text)
            if digit_ratio > 0.5:
                score -= 0.1

        return max(0.0, min(1.0, score))


class ChunkDeduplicator:
    """Detect and remove near-duplicate chunks using SimHash."""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self._hashes: Dict[int, str] = {}  # simhash -> chunk_id

    def _simhash(self, text: str, hashbits: int = 64) -> int:
        """Compute SimHash for text."""
        tokens = text.lower().split()
        v = [0] * hashbits

        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(hashbits):
                if token_hash & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1

        fingerprint = 0
        for i in range(hashbits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    def _hamming_distance(self, a: int, b: int, bits: int = 64) -> int:
        return bin(a ^ b).count('1')

    def is_duplicate(self, chunk: Chunk) -> bool:
        """Check if chunk is near-duplicate of an existing one."""
        h = self._simhash(chunk.content)
        for existing_hash, cid in self._hashes.items():
            distance = self._hamming_distance(h, existing_hash)
            similarity = 1.0 - (distance / 64.0)
            if similarity >= self.threshold:
                return True
        self._hashes[h] = chunk.chunk_id
        return False

    def clear(self):
        self._hashes.clear()


class DocumentChunker:
    """Chunk documents — v2 with page mode, quality scoring, dedup."""

    def __init__(self, config: Optional[ChunkingConfig] = None,
                 page_mode: bool = False,
                 min_quality: float = 0.3,
                 deduplicate: bool = False):
        self.config = config or ChunkingConfig()
        self.page_mode = page_mode
        self.min_quality = min_quality
        self.deduplicate = deduplicate
        self._sentence_endings = re.compile(r"(?<=[.!?])\s+")
        self._paragraph_breaks = re.compile(r"\n\s*\n")
        self._scorer = ChunkQualityScorer()
        self._deduplicator = ChunkDeduplicator() if deduplicate else None

    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """Chunk a parsed document into retrievable pieces."""
        if self.page_mode and document.pages:
            chunks = self._chunk_pages(document)
        else:
            chunks = self._chunk_text_mode(document)

        # Quality gate
        if self.min_quality > 0:
            before = len(chunks)
            chunks = [c for c in chunks if c.quality_score >= self.min_quality]
            rejected = before - len(chunks)
            if rejected > 0:
                logger.info(f"Quality gate rejected {rejected} chunks from {document.filename}")

        # Deduplication
        if self._deduplicator:
            before = len(chunks)
            chunks = [c for c in chunks if not self._deduplicator.is_duplicate(c)]
            deduped = before - len(chunks)
            if deduped > 0:
                logger.info(f"Dedup removed {deduped} chunks from {document.filename}")

        # Set chunk indices and link prev/next
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            if i > 0:
                chunk.prev_chunk_id = chunks[i - 1].chunk_id
                chunks[i - 1].next_chunk_id = chunk.chunk_id

        logger.info(f"Created {len(chunks)} chunks from {document.filename}")
        return chunks

    def _chunk_pages(self, document: ParsedDocument) -> List[Chunk]:
        """Page-level chunking — each page is a retrieval unit."""
        chunks = []
        for page in document.pages:
            if not page.text.strip():
                continue

            chunk = Chunk(
                chunk_id="",
                doc_id=document.doc_id,
                content=page.text,
                source_document=document.filename,
                is_page=True,
                page_number=page.page_number,
                metadata={"page": page.page_number},
            )
            chunk.quality_score = self._scorer.score(chunk)
            chunks.append(chunk)

            # Page tables as separate chunks
            for table in page.tables:
                table_chunk = self._make_table_chunk(
                    table, document.doc_id, document.filename, page.page_number
                )
                if table_chunk:
                    chunks.extend(table_chunk)

        return chunks

    def _chunk_text_mode(self, document: ParsedDocument) -> List[Chunk]:
        """Standard text chunking with overlap and boundary detection."""
        chunks = []

        # Chunk main text
        text_chunks = self._chunk_text(
            document.content,
            doc_id=document.doc_id,
            source=document.filename,
        )
        chunks.extend(text_chunks)

        # Chunk tables separately
        for i, table in enumerate(document.tables):
            table_chunks = self._make_table_chunk(
                table, document.doc_id, document.filename, i
            )
            chunks.extend(table_chunks)

        return chunks

    def _chunk_text(self, text: str, doc_id: str = "", source: str = "") -> List[Chunk]:
        """Chunk text with overlap and boundary respect."""
        if not text.strip():
            return []

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        min_size = self.config.min_chunk_size

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
                if len(current_text) >= min_size:
                    chunk = Chunk(
                        chunk_id="", doc_id=doc_id,
                        content=current_text.strip(),
                        source_document=source,
                        start_char=current_start,
                        end_char=current_start + len(current_text),
                    )
                    chunk.quality_score = self._scorer.score(chunk)
                    chunks.append(chunk)

                if overlap > 0 and current_text:
                    overlap_text = current_text[-overlap:]
                    sentences = self._sentence_endings.split(overlap_text)
                    if len(sentences) > 1:
                        overlap_text = sentences[-1]
                    current_text = overlap_text + "\n\n" + para
                else:
                    current_text = para
                current_start = char_offset

            char_offset += len(para) + 2

        if len(current_text) >= min_size:
            chunk = Chunk(
                chunk_id="", doc_id=doc_id,
                content=current_text.strip(),
                source_document=source,
                start_char=current_start,
                end_char=current_start + len(current_text),
            )
            chunk.quality_score = self._scorer.score(chunk)
            chunks.append(chunk)
        elif current_text.strip() and chunks:
            chunks[-1].content += "\n\n" + current_text.strip()

        return chunks

    def _make_table_chunk(self, table: Dict, doc_id: str, source: str,
                          index: int) -> List[Chunk]:
        """Create chunks from a table, preserving structure."""
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        if not rows:
            return []

        header_line = " | ".join(str(h) for h in headers) if headers else ""
        row_lines = [" | ".join(str(c) for c in row) for row in rows]

        full_table = header_line + "\n" + "\n".join(row_lines) if header_line else "\n".join(row_lines)

        if len(full_table) <= self.config.chunk_size:
            chunk = Chunk(
                chunk_id="", doc_id=doc_id,
                content=full_table, source_document=source,
                is_table=True,
                page_number=table.get("page", 0),
                metadata={
                    "table_index": index,
                    "row_count": len(rows),
                    "columns": headers,
                    "page": table.get("page"),
                },
            )
            chunk.quality_score = self._scorer.score(chunk)
            return [chunk]

        # Split large tables
        chunks = []
        batch_size = max(1, self.config.chunk_size // (len(header_line) + 50))
        for i in range(0, len(row_lines), batch_size):
            batch = row_lines[i:i + batch_size]
            chunk_content = header_line + "\n" + "\n".join(batch) if header_line else "\n".join(batch)
            chunk = Chunk(
                chunk_id="", doc_id=doc_id,
                content=chunk_content, source_document=source,
                is_table=True,
                page_number=table.get("page", 0),
                metadata={
                    "table_index": index,
                    "row_range": f"{i}-{i + len(batch)}",
                    "columns": headers,
                },
            )
            chunk.quality_score = self._scorer.score(chunk)
            chunks.append(chunk)

        return chunks
