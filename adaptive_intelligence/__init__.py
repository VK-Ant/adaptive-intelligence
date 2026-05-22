"""
Adaptive Intelligence v2 — Self-Improving Retrieval Orchestration Framework

Drop documents. Ask questions. The system learns how to retrieve better over time.

v2: Vectorless mode, output formats, user feedback, crash recovery,
    incremental ingestion, SQL connector, page citations.

Usage:
    from adaptive_intelligence import AdaptiveAI

    engine = AdaptiveAI()
    engine.ingest("./documents")
    response = engine.ask("What are the key risks?")

    # v2: vectorless
    engine = AdaptiveAI(vectorless=True)

    # v2: output formats
    response = engine.ask("Extract vendors", output_format="json")

    # v2: feedback
    engine.feedback(response.query_id, "good")
"""

__version__ = "2.0.1"
__author__ = "Venkatkumar Rajan"

from adaptive_intelligence.core.engine import AdaptiveAI
from adaptive_intelligence.core.response import AdaptiveResponse

__all__ = ["AdaptiveAI", "AdaptiveResponse"]
