"""
Demo 5: MCP Client
====================

Connects to the adaptive-intelligence MCP server.

Run: python demo_mcp_server.py (in terminal 1)
     python demo_mcp_client.py (in terminal 2)
"""

import json
import time
import urllib.request


MCP_SERVER = "http://localhost:8080"


def mcp_call(method, params=None):
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": 1,
    }
    data = json.dumps(request).encode()
    req = urllib.request.Request(
        MCP_SERVER, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def wait_for_server():
    """Wait until MCP server is ready."""
    print("Connecting to MCP server...")
    for i in range(15):
        try:
            mcp_call("tools/list")
            print("Connected!\n")
            return True
        except Exception:
            if i == 0:
                print("Server is starting up. Please wait...")
            print(f"  [{i+1}/15] Server not ready yet. Retrying in 5 seconds...")
            time.sleep(5)
    return False


def main():
    print("=" * 60)
    print("DEMO 5: MCP Client")
    print("=" * 60)

    if not wait_for_server():
        print("\nCould not connect to MCP server.")
        print("Make sure the server is running: python demo_mcp_server.py")
        return

    # List tools
    print("--- Available tools ---")
    response = mcp_call("tools/list")
    tools = response.get("result", {}).get("tools", [])
    for tool in tools:
        print(f"  {tool['name']}: {tool['description'][:80]}")

    # Send queries
    queries = [
        "What is Q3 revenue?",
        "What are the supply chain risks?",
        "Metformin dosage for Type 2 Diabetes?",
        "Compare Q2 vs Q3 financial performance",
        "Empagliflozin cardiovascular trial results?",
    ]

    print("\n--- Sending queries via MCP ---")
    for query in queries:
        print(f"\nQ: {query}")
        response = mcp_call("tools/call", {
            "name": "adaptive_intelligence_search",
            "arguments": {"query": query},
        })

        result = response.get("result", {})
        content = result.get("content", [])
        metadata = result.get("metadata", {})

        if content:
            answer = content[0].get("text", "")
            print(f"A: {answer[:300]}")
            print(f"   Confidence: {metadata.get('confidence', 0):.0%}")
            print(f"   Strategy: {metadata.get('strategy', 'unknown')}")

    print("\n--- Done ---")
    print("All queries handled via MCP protocol with RL-optimized retrieval.")


if __name__ == "__main__":
    main()
