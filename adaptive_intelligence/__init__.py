"""
Adaptive Intelligence v4 — Self-Improving Retrieval Orchestration Framework

v4: Context engineering, MCP integration, agentic workflow,
persistent memory, incremental learning, tool registry.

Usage:
    from adaptive_intelligence import AdaptiveAI

    engine = AdaptiveAI()
    engine.ingest("./documents")
    response = engine.ask("What are the key risks?")

    # v4: Add tools
    engine.add_tool("financial", server="http://localhost:8081")

    # v4: Agentic mode
    response = engine.ask("Deep research on risks", mode="agentic")

    # v4: Memory persists across sessions
    engine.remember("user_preference", "prefers detailed analysis")

    # v4: MCP server
    engine.serve_mcp(port=8080)
"""

__version__ = "4.0.0"
__author__ = "Venkatkumar Rajan"

from adaptive_intelligence.core.engine import AdaptiveAI
from adaptive_intelligence.core.response import AdaptiveResponse

__all__ = ["AdaptiveAI", "AdaptiveResponse"]
