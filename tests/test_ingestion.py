"""Tests for the Ingestion Pipeline (parser + chunker + engine)."""

import os
import pytest
import tempfile
from pathlib import Path

from adaptive_intelligence.ingestion.parser import DocumentParser, ParsedDocument
from adaptive_intelligence.ingestion.chunker import DocumentChunker, Chunk
from adaptive_intelligence.ingestion.engine import IngestionEngine
from adaptive_intelligence.core.config import ChunkingConfig


@pytest.fixture
def parser():
    return DocumentParser()


@pytest.fixture
def chunker():
    return DocumentChunker(ChunkingConfig(chunk_size=200, chunk_overlap=30))


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def create_text_file(directory, name, content):
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        f.write(content)
    return path


class TestDocumentParser:
    def test_parse_text_file(self, parser, temp_dir):
        path = create_text_file(temp_dir, "test.txt", "Hello world.\nThis is a test document.")
        doc = parser.parse(path)
        assert isinstance(doc, ParsedDocument)
        assert "Hello world" in doc.content
        assert doc.file_type == "text"

    def test_parse_markdown(self, parser, temp_dir):
        content = "# Title\n\nSome content here.\n\n## Section 2\n\nMore content."
        path = create_text_file(temp_dir, "test.md", content)
        doc = parser.parse(path)
        assert doc.file_type == "markdown"
        assert len(doc.sections) >= 1

    def test_parse_csv(self, parser, temp_dir):
        content = "Name,Revenue,Year\nAcme,500,2024\nBeta,300,2024"
        path = create_text_file(temp_dir, "data.csv", content)
        doc = parser.parse(path)
        assert doc.file_type == "csv"
        assert len(doc.tables) == 1
        assert doc.tables[0]["headers"] == ["Name", "Revenue", "Year"]
        assert len(doc.tables[0]["rows"]) == 2

    def test_parse_json(self, parser, temp_dir):
        content = '{"name": "Acme", "revenue": 500}'
        path = create_text_file(temp_dir, "data.json", content)
        doc = parser.parse(path)
        assert doc.file_type == "json"
        assert "Acme" in doc.content

    def test_parse_html(self, parser, temp_dir):
        content = "<html><body><h1>Title</h1><p>Content here.</p></body></html>"
        path = create_text_file(temp_dir, "page.html", content)
        doc = parser.parse(path)
        assert doc.file_type == "html"
        assert "Title" in doc.content
        assert "<h1>" not in doc.content  # Tags stripped

    def test_unsupported_format(self, parser, temp_dir):
        path = create_text_file(temp_dir, "test.xyz", "content")
        with pytest.raises(ValueError, match="Unsupported"):
            parser.parse(path)

    def test_file_not_found(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.txt")

    def test_parse_directory(self, parser, temp_dir):
        create_text_file(temp_dir, "a.txt", "File A content")
        create_text_file(temp_dir, "b.txt", "File B content")
        create_text_file(temp_dir, "c.csv", "col1,col2\n1,2")
        docs = parser.parse_directory(temp_dir)
        assert len(docs) == 3

    def test_metadata(self, parser, temp_dir):
        path = create_text_file(temp_dir, "meta.txt", "content")
        doc = parser.parse(path)
        assert "file_size" in doc.metadata
        assert "extension" in doc.metadata
        assert doc.metadata["extension"] == ".txt"


class TestDocumentChunker:
    def test_basic_chunking(self, chunker):
        doc = ParsedDocument(
            doc_id="d1", filename="test.txt", filepath="test.txt",
            content="Paragraph one with enough text to form a chunk.\n\n"
                    "Paragraph two with additional information.\n\n"
                    "Paragraph three concluding the document.",
            file_type="text",
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_unique(self, chunker):
        doc = ParsedDocument(
            doc_id="d1", filename="test.txt", filepath="test.txt",
            content="A " * 500, file_type="text",
        )
        chunks = chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_table_chunking(self, chunker):
        doc = ParsedDocument(
            doc_id="d1", filename="data.csv", filepath="data.csv",
            content="header content", file_type="csv",
            tables=[{
                "headers": ["Name", "Value"],
                "rows": [["A", "1"], ["B", "2"]],
            }],
        )
        chunks = chunker.chunk_document(doc)
        table_chunks = [c for c in chunks if c.is_table]
        assert len(table_chunks) >= 1

    def test_empty_content(self, chunker):
        doc = ParsedDocument(
            doc_id="d1", filename="empty.txt", filepath="empty.txt",
            content="", file_type="text",
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 0

    def test_respects_chunk_size(self):
        chunker = DocumentChunker(ChunkingConfig(chunk_size=200, chunk_overlap=10, min_chunk_size=20))
        # Use paragraph breaks so the chunker can split
        paragraphs = ["This is paragraph number %d with enough text." % i for i in range(30)]
        content = "\n\n".join(paragraphs)
        doc = ParsedDocument(
            doc_id="d1", filename="test.txt", filepath="test.txt",
            content=content,
            file_type="text",
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1  # Should split into multiple chunks


class TestIngestionEngine:
    def test_ingest_file(self, temp_dir):
        paragraphs = ["Paragraph %d about revenue and profits with details." % i for i in range(10)]
        path = create_text_file(temp_dir, "doc.txt", "\n\n".join(paragraphs))
        engine = IngestionEngine()
        stats = engine.ingest(path)
        assert stats.successful == 1
        assert stats.total_chunks >= 1

    def test_ingest_directory(self, temp_dir):
        create_text_file(temp_dir, "a.txt", "Document A content " * 20)
        create_text_file(temp_dir, "b.txt", "Document B content " * 20)
        engine = IngestionEngine()
        stats = engine.ingest(temp_dir)
        assert stats.successful == 2
        assert stats.total_chunks >= 2

    def test_get_chunks(self, temp_dir):
        create_text_file(temp_dir, "doc.txt", "Content " * 50)
        engine = IngestionEngine()
        engine.ingest(temp_dir)
        chunks = engine.get_chunks()
        assert len(chunks) >= 1

    def test_clear(self, temp_dir):
        create_text_file(temp_dir, "doc.txt", "Content " * 50)
        engine = IngestionEngine()
        engine.ingest(temp_dir)
        assert len(engine.get_chunks()) > 0
        engine.clear()
        assert len(engine.get_chunks()) == 0
