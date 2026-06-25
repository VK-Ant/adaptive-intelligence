"""
Demo 4: MCP Server
===================

Serves adaptive-intelligence as an MCP tool.
Any MCP client (Claude, Cursor, custom apps) can connect.

Run: python demo_mcp_server.py
Then in another terminal: python demo_mcp_client.py

Press Ctrl+C to stop the server.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adaptive_intelligence import AdaptiveAI


def main():
    print("=" * 60)
    print("DEMO 4: MCP Server")
    print("=" * 60)

    # Using llm_backend="none" for instant startup
    # For synthesized answers, change to:
    #   api_key="gsk_...", base_url="https://api.groq.com/openai/v1" (Groq free)
    engine = AdaptiveAI(
        llm_backend="none",
        vectorless=True,
        storage_dir="./demo_state_mcp",
        log_level="ERROR",
    )

    print("Ingesting documents...")
    engine.ingest("./data/financial")
    engine.ingest("./data/healthcare")

    # Warmup - run one query to initialize everything
    print("Warming up...")
    engine.ask("test query")
    print(f"Ready. Docs ingested. System warmed up.")

    # Start MCP server
    print("\nStarting MCP server on port 8080...")
    print("Any MCP client can now connect to http://localhost:8080")
    print()
    print("Test with: python demo_mcp_client.py (in another terminal)")
    print("Press Ctrl+C to stop.")
    print("-" * 60)

    engine.serve_mcp(port=8080)


if __name__ == "__main__":
    main()
