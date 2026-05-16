"""
Adaptive Intelligence — Self-Improving Retrieval Orchestration Framework

Drop documents. Ask questions. The system learns how to retrieve better over time.

Usage:
    from adaptive_intelligence import AdaptiveAI

    engine = AdaptiveAI()
    engine.ingest("./documents")
    response = engine.ask("What are the operational risks?")
    print(response.answer)
    print(response.confidence)
    print(response.citations)
"""

__version__ = "1.0.0"
__author__ = "Venkatkumar Rajan"

from adaptive_intelligence.core.engine import AdaptiveAI
from adaptive_intelligence.core.response import AdaptiveResponse

__all__ = ["AdaptiveAI", "AdaptiveResponse"]
