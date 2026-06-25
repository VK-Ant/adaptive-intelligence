"""
Demo 5: MCP Client
====================

Connects to the adaptive-intelligence MCP server
and sends queries as an MCP client.

Run: python demo_mcp_server.py (in terminal 1)
     python demo_mcp_client.py (in terminal 2)
"""

import json
import urllib.request


MCP_SERVER = "http://localhost:8080"


def mcp_call(method, params=None):
    """Send an MCP request to the server."""
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

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main():
    print("=" * 60)
    print("DEMO 5: MCP Client")
    print("=" * 60)
    print(f"Connecting to {MCP_SERVER}...")

    # List available tools
    print("\n--- Available tools ---")
    try:
        response = mcp_call("tools/list")
        tools = response.get("result", {}).get("tools", [])
        for tool in tools:
            print(f"  {tool['name']}: {tool['description']}")
    except Exception as e:
        print(f"Cannot connect to MCP server: {e}")
        print("Start the server first: python demo_mcp_server.py")
        return

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
        else:
            print(f"   Error: {result}")

    print("\n--- Done ---")
    print("The MCP server handled all queries using RL-optimized retrieval.")
    print("Any MCP client can connect the same way.")


if __name__ == "__main__":
    main()
