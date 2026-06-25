"""
Demo 4: MCP Server
===================

Serves adaptive-intelligence as an MCP tool.
Any MCP client (Claude, Cursor, custom apps) can connect
and use your retrieval as a tool.

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

    # Setup engine
    engine = AdaptiveAI(
        llm_backend="none", vectorless=True,
        storage_dir="./demo_state_mcp",
        log_level="ERROR",
    )

    engine.ingest("./data/financial")
    engine.ingest("./data/healthcare")
    print(f"Ingested docs. Ready to serve.")

    # Start MCP server
    print("\nStarting MCP server on port 8080...")
    print("Any MCP client can now connect to http://localhost:8080")
    print()
    print("Test with: python demo_mcp_client.py (in another terminal)")
    print("Or connect from Claude Desktop, Cursor, or any MCP client.")
    print()
    print("Press Ctrl+C to stop.")
    print("-" * 60)

    engine.serve_mcp(port=8080)


if __name__ == "__main__":
    main()
