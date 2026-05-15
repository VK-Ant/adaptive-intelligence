"""Security & Governance — Audit trail, dependency verification, privacy.

Zero network by default. Full traceability. Every decision logged.
"""

import hashlib
import json
import logging
import time
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """A single audit trail entry."""
    timestamp: float
    event_type: str  # "query", "retrieval", "rl_decision", "evaluation", "ingestion"
    details: Dict[str, Any] = field(default_factory=dict)
    query_id: str = ""

    @property
    def formatted_time(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def __repr__(self) -> str:
        return f"[{self.formatted_time}] {self.event_type}: {json.dumps(self.details, default=str)}"


class AuditTrail:
    """Full audit trail for every system decision."""

    def __init__(self, persist_dir: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self._persist_dir = persist_dir
        self._entries: List[AuditEntry] = []
        self._query_trails: Dict[str, List[AuditEntry]] = {}

    def log(self, event_type: str, details: Dict[str, Any], query_id: str = ""):
        """Log an audit event."""
        if not self.enabled:
            return

        entry = AuditEntry(
            timestamp=time.time(),
            event_type=event_type,
            details=details,
            query_id=query_id,
        )
        self._entries.append(entry)

        if query_id:
            if query_id not in self._query_trails:
                self._query_trails[query_id] = []
            self._query_trails[query_id].append(entry)

        logger.debug(str(entry))

    def get_query_trail(self, query_id: str) -> List[AuditEntry]:
        """Get the complete audit trail for a specific query."""
        return self._query_trails.get(query_id, [])

    def get_recent(self, n: int = 50) -> List[AuditEntry]:
        """Get the most recent audit entries."""
        return self._entries[-n:]

    def export(self, filepath: Optional[str] = None) -> str:
        """Export audit trail to JSON."""
        data = [
            {
                "timestamp": e.formatted_time,
                "event_type": e.event_type,
                "query_id": e.query_id,
                "details": e.details,
            }
            for e in self._entries
        ]
        output = json.dumps(data, indent=2, default=str)

        if filepath:
            with open(filepath, "w") as f:
                f.write(output)
            logger.info(f"Audit trail exported to {filepath}")

        return output

    def display_query_trail(self, query_id: str) -> str:
        """Human-readable audit trail for a query."""
        entries = self.get_query_trail(query_id)
        if not entries:
            return f"No audit trail found for query: {query_id}"

        lines = [f"Audit Trail for Query: {query_id}", "=" * 60]
        for entry in entries:
            lines.append(str(entry))
        return "\n".join(lines)

    def clear(self):
        """Clear the audit trail."""
        self._entries.clear()
        self._query_trails.clear()


class DependencyVerifier:
    """Verify integrity of installed dependencies."""

    # Core dependencies with expected packages
    CORE_DEPENDENCIES = [
        "chromadb",
    ]

    OPTIONAL_DEPENDENCIES = {
        "pdf": ["fitz", "pdfplumber"],
        "docx": ["docx"],
        "xlsx": ["openpyxl"],
        "pptx": ["pptx"],
        "ocr": ["pytesseract", "PIL"],
        "huggingface": ["transformers", "torch"],
    }

    @classmethod
    def verify_core(cls) -> Dict[str, Any]:
        """Verify core dependencies are available."""
        results = {}
        for dep in cls.CORE_DEPENDENCIES:
            try:
                __import__(dep)
                results[dep] = {"status": "ok", "available": True}
            except ImportError:
                results[dep] = {"status": "missing", "available": False}
        return results

    @classmethod
    def verify_optional(cls) -> Dict[str, Dict[str, Any]]:
        """Verify optional dependencies."""
        results = {}
        for category, deps in cls.OPTIONAL_DEPENDENCIES.items():
            available = False
            for dep in deps:
                try:
                    __import__(dep)
                    available = True
                    break
                except ImportError:
                    continue
            results[category] = {"available": available, "packages": deps}
        return results

    @classmethod
    def full_report(cls) -> str:
        """Generate a full dependency report."""
        core = cls.verify_core()
        optional = cls.verify_optional()

        lines = ["Dependency Report", "=" * 40, "", "Core Dependencies:"]
        for dep, info in core.items():
            status = "✓" if info["available"] else "✗ MISSING"
            lines.append(f"  {dep}: {status}")

        lines.extend(["", "Optional Dependencies:"])
        for cat, info in optional.items():
            status = "✓" if info["available"] else "✗ not installed"
            lines.append(f"  {cat}: {status} ({', '.join(info['packages'])})")

        lines.extend(["", f"Python: {sys.version.split()[0]}"])

        return "\n".join(lines)


class SecurityManager:
    """Manages security settings and PII detection."""

    # Simple PII patterns
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }

    def __init__(self, security_level: str = "standard"):
        self.security_level = security_level

    def scan_for_pii(self, text: str) -> List[Dict[str, str]]:
        """Scan text for potential PII."""
        import re
        findings = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            for match in matches:
                findings.append({"type": pii_type, "value": match[:4] + "***"})
        return findings

    def check_network_safety(self, url: str) -> bool:
        """Check if a URL is safe to access."""
        # In maximum security, block all network
        if self.security_level == "maximum":
            return False
        # In high security, only allow whitelisted domains
        if self.security_level == "high":
            allowed = {"api.openai.com", "api.anthropic.com", "localhost"}
            from urllib.parse import urlparse
            domain = urlparse(url).hostname
            return domain in allowed
        return True

    def hash_content(self, content: str) -> str:
        """Create SHA-256 hash of content for integrity verification."""
        return hashlib.sha256(content.encode()).hexdigest()
