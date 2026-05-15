"""Document parsing for multiple file formats."""

import os
import csv
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Structured representation of a parsed document."""
    doc_id: str
    filename: str
    filepath: str
    content: str
    file_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    sections: List[Dict[str, str]] = field(default_factory=list)
    page_count: int = 0
    char_count: int = 0
    parsed_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.char_count = len(self.content)
        if not self.doc_id:
            self.doc_id = hashlib.md5(
                f"{self.filepath}:{self.content[:500]}".encode()
            ).hexdigest()[:12]


class DocumentParser:
    """Universal document parser supporting multiple formats."""

    SUPPORTED_EXTENSIONS = {
        ".txt": "text",
        ".md": "markdown",
        ".csv": "csv",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".pdf": "pdf",
        ".docx": "docx",
        ".xlsx": "xlsx",
        ".pptx": "pptx",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".tiff": "image",
        ".rtf": "rtf",
        ".xml": "xml",
    }

    def __init__(self):
        self._parsers = {
            "text": self._parse_text,
            "markdown": self._parse_text,
            "csv": self._parse_csv,
            "json": self._parse_json,
            "html": self._parse_html,
            "pdf": self._parse_pdf,
            "docx": self._parse_docx,
            "xlsx": self._parse_xlsx,
            "pptx": self._parse_pptx,
            "image": self._parse_image,
            "rtf": self._parse_text,
            "xml": self._parse_xml,
        }

    def parse(self, filepath: str) -> ParsedDocument:
        """Parse a single document."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = path.suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(ext)
        if not file_type:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        parser_fn = self._parsers[file_type]
        logger.info(f"Parsing {path.name} as {file_type}")

        try:
            doc = parser_fn(str(path))
            doc.filename = path.name
            doc.filepath = str(path)
            doc.file_type = file_type
            doc.metadata["file_size"] = path.stat().st_size
            doc.metadata["extension"] = ext
            doc.metadata["modified"] = datetime.fromtimestamp(
                path.stat().st_mtime
            ).isoformat()
            return doc
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            raise

    def parse_directory(self, directory: str, recursive: bool = True) -> List[ParsedDocument]:
        """Parse all supported documents in a directory."""
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        documents = []
        pattern = "**/*" if recursive else "*"

        for file_path in sorted(path.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.parse(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")

        logger.info(f"Parsed {len(documents)} documents from {directory}")
        return documents

    def _parse_text(self, filepath: str) -> ParsedDocument:
        """Parse plain text or markdown files."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        sections = []
        current_section = {"title": "Main", "content": ""}
        for line in content.split("\n"):
            if line.startswith("#"):
                if current_section["content"].strip():
                    sections.append(current_section)
                title = line.lstrip("#").strip()
                current_section = {"title": title, "content": ""}
            else:
                current_section["content"] += line + "\n"
        if current_section["content"].strip():
            sections.append(current_section)

        return ParsedDocument(
            doc_id="",
            filename="",
            filepath=filepath,
            content=content,
            file_type="text",
            sections=sections,
        )

    def _parse_csv(self, filepath: str) -> ParsedDocument:
        """Parse CSV files into structured content."""
        rows = []
        headers = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    headers = row
                else:
                    rows.append(row)

        content_lines = [", ".join(headers)]
        for row in rows:
            content_lines.append(", ".join(row))
        content = "\n".join(content_lines)

        tables = [{"headers": headers, "rows": rows, "source": filepath}]

        return ParsedDocument(
            doc_id="",
            filename="",
            filepath=filepath,
            content=content,
            file_type="csv",
            tables=tables,
            metadata={"row_count": len(rows), "column_count": len(headers), "columns": headers},
        )

    def _parse_json(self, filepath: str) -> ParsedDocument:
        """Parse JSON files."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        content = json.dumps(data, indent=2)
        return ParsedDocument(
            doc_id="",
            filename="",
            filepath=filepath,
            content=content,
            file_type="json",
            metadata={"json_type": type(data).__name__},
        )

    def _parse_html(self, filepath: str) -> ParsedDocument:
        """Parse HTML files, stripping tags."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        import re
        # Strip script and style blocks
        clean = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        # Strip tags
        clean = re.sub(r"<[^>]+>", " ", clean)
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean).strip()

        return ParsedDocument(
            doc_id="",
            filename="",
            filepath=filepath,
            content=clean,
            file_type="html",
        )

    def _parse_pdf(self, filepath: str) -> ParsedDocument:
        """Parse PDF files using PyMuPDF or pdfplumber."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            pages = []
            tables = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                pages.append(text)
                # Extract tables if available
                page_tables = page.find_tables()
                for table in page_tables:
                    table_data = table.extract()
                    if table_data and len(table_data) > 1:
                        tables.append({
                            "headers": table_data[0],
                            "rows": table_data[1:],
                            "page": page_num + 1,
                            "source": filepath,
                        })
            doc.close()
            content = "\n\n".join(pages)
            return ParsedDocument(
                doc_id="",
                filename="",
                filepath=filepath,
                content=content,
                file_type="pdf",
                tables=tables,
                page_count=len(pages),
            )
        except ImportError:
            pass

        try:
            import pdfplumber
            pages = []
            tables = []
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    pages.append(text)
                    for table in page.extract_tables():
                        if table and len(table) > 1:
                            tables.append({
                                "headers": table[0],
                                "rows": table[1:],
                                "page": page_num + 1,
                                "source": filepath,
                            })
            content = "\n\n".join(pages)
            return ParsedDocument(
                doc_id="",
                filename="",
                filepath=filepath,
                content=content,
                file_type="pdf",
                tables=tables,
                page_count=len(pages),
            )
        except ImportError:
            raise ImportError(
                "PDF parsing requires PyMuPDF or pdfplumber. "
                "Install with: pip install PyMuPDF or pip install pdfplumber"
            )

    def _parse_docx(self, filepath: str) -> ParsedDocument:
        """Parse Word documents."""
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(filepath)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            content = "\n\n".join(paragraphs)

            tables = []
            for table in doc.tables:
                rows = []
                for row in table.rows:
                    rows.append([cell.text for cell in row.cells])
                if rows:
                    tables.append({
                        "headers": rows[0],
                        "rows": rows[1:],
                        "source": filepath,
                    })

            return ParsedDocument(
                doc_id="",
                filename="",
                filepath=filepath,
                content=content,
                file_type="docx",
                tables=tables,
            )
        except ImportError:
            raise ImportError("DOCX parsing requires python-docx. Install with: pip install python-docx")

    def _parse_xlsx(self, filepath: str) -> ParsedDocument:
        """Parse Excel files."""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(filepath, data_only=True)
            content_parts = []
            tables = []

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows = []
                for row in ws.iter_rows(values_only=True):
                    row_data = [str(cell) if cell is not None else "" for cell in row]
                    rows.append(row_data)

                if rows:
                    content_parts.append(f"Sheet: {sheet_name}")
                    for row in rows:
                        content_parts.append(", ".join(row))
                    tables.append({
                        "sheet": sheet_name,
                        "headers": rows[0] if rows else [],
                        "rows": rows[1:] if len(rows) > 1 else [],
                        "source": filepath,
                    })

            return ParsedDocument(
                doc_id="",
                filename="",
                filepath=filepath,
                content="\n".join(content_parts),
                file_type="xlsx",
                tables=tables,
                metadata={"sheet_count": len(wb.sheetnames), "sheets": wb.sheetnames},
            )
        except ImportError:
            raise ImportError("XLSX parsing requires openpyxl. Install with: pip install openpyxl")

    def _parse_pptx(self, filepath: str) -> ParsedDocument:
        """Parse PowerPoint files."""
        try:
            from pptx import Presentation
            prs = Presentation(filepath)
            slides = []
            for i, slide in enumerate(prs.slides):
                texts = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            text = paragraph.text.strip()
                            if text:
                                texts.append(text)
                    if shape.has_table:
                        for row in shape.table.rows:
                            row_text = [cell.text for cell in row.cells]
                            texts.append(" | ".join(row_text))
                if texts:
                    slides.append(f"Slide {i+1}:\n" + "\n".join(texts))

            return ParsedDocument(
                doc_id="",
                filename="",
                filepath=filepath,
                content="\n\n".join(slides),
                file_type="pptx",
                page_count=len(prs.slides),
            )
        except ImportError:
            raise ImportError("PPTX parsing requires python-pptx. Install with: pip install python-pptx")

    def _parse_image(self, filepath: str) -> ParsedDocument:
        """Parse images via OCR."""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(filepath)
            content = pytesseract.image_to_string(img)
            return ParsedDocument(
                doc_id="",
                filename="",
                filepath=filepath,
                content=content,
                file_type="image",
                metadata={"width": img.width, "height": img.height, "mode": img.mode},
            )
        except ImportError:
            logger.warning(f"OCR not available for {filepath}. Install pytesseract and Pillow.")
            return ParsedDocument(
                doc_id="",
                filename="",
                filepath=filepath,
                content=f"[Image file: {filepath}. OCR not available.]",
                file_type="image",
            )

    def _parse_xml(self, filepath: str) -> ParsedDocument:
        """Parse XML files."""
        import xml.etree.ElementTree as ET
        tree = ET.parse(filepath)
        root = tree.getroot()

        def extract_text(element, depth=0):
            texts = []
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
            if element.text and element.text.strip():
                texts.append(f"{'  ' * depth}{tag}: {element.text.strip()}")
            for child in element:
                texts.extend(extract_text(child, depth + 1))
            return texts

        content = "\n".join(extract_text(root))
        return ParsedDocument(
            doc_id="",
            filename="",
            filepath=filepath,
            content=content,
            file_type="xml",
        )
