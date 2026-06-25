"""MCP Integration — v4 addition.

Two modes:
1. MCP Server: Your library serves retrieval as an MCP tool
2. Tool Registry: Connect external MCP tools (financial, medical, web search)

The RL policy learns which tools to call per query type.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool call."""
    tool_name: str
    result: str
    latency: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class ToolConfig:
    """Configuration for a registered tool."""
    name: str
    description: str
    tool_type: str  # "mcp", "function", "api"
    server_url: Optional[str] = None
    function: Optional[Callable] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    enabled: bool = True
    call_count: int = 0
    success_count: int = 0
    avg_latency: float = 0.0


class ToolRegistry:
    """Registry of external tools the system can call.

    The RL policy learns which tools to call per query type.
    Tools can be MCP servers, Python functions, or API endpoints.
    """

    def __init__(self):
        self._tools: Dict[str, ToolConfig] = {}
        self._call_history: List[Dict[str, Any]] = []

    def add_tool(self, name: str, description: str = "",
                 server: str = None, function: Callable = None,
                 api_endpoint: str = None, api_key: str = None):
        """Register a tool.

        Args:
            name: Tool identifier (e.g., "financial", "medical")
            description: What the tool does
            server: MCP server URL
            function: Python callable
            api_endpoint: REST API URL
            api_key: API authentication key
        """
        if server:
            tool_type = "mcp"
        elif function:
            tool_type = "function"
        elif api_endpoint:
            tool_type = "api"
        else:
            raise ValueError("Provide server, function, or api_endpoint")

        self._tools[name] = ToolConfig(
            name=name, description=description or name,
            tool_type=tool_type, server_url=server,
            function=function, api_endpoint=api_endpoint,
            api_key=api_key,
        )
        logger.info(f"Tool registered: {name} ({tool_type})")

    def remove_tool(self, name: str):
        if name in self._tools:
            del self._tools[name]

    def call_tool(self, name: str, query: str,
                  params: Dict = None) -> ToolResult:
        """Call a registered tool."""
        tool = self._tools.get(name)
        if not tool or not tool.enabled:
            return ToolResult(
                tool_name=name, result="", success=False,
                error=f"Tool '{name}' not found or disabled",
            )

        start = time.time()
        try:
            if tool.tool_type == "function":
                result = self._call_function(tool, query, params)
            elif tool.tool_type == "mcp":
                result = self._call_mcp(tool, query, params)
            elif tool.tool_type == "api":
                result = self._call_api(tool, query, params)
            else:
                result = ToolResult(tool_name=name, result="", success=False,
                                    error="Unknown tool type")

            result.latency = time.time() - start
            tool.call_count += 1
            if result.success:
                tool.success_count += 1
            tool.avg_latency = (
                (tool.avg_latency * (tool.call_count - 1) + result.latency)
                / tool.call_count
            )

            self._call_history.append({
                "tool": name, "query": query[:100],
                "success": result.success, "latency": result.latency,
                "timestamp": time.time(),
            })

            return result

        except Exception as e:
            logger.error(f"Tool call failed: {name}: {e}")
            return ToolResult(
                tool_name=name, result="", success=False,
                error=str(e), latency=time.time() - start,
            )

    def _call_function(self, tool: ToolConfig, query: str,
                       params: Dict = None) -> ToolResult:
        """Call a Python function tool."""
        result = tool.function(query, **(params or {}))
        return ToolResult(
            tool_name=tool.name,
            result=str(result) if not isinstance(result, str) else result,
            success=True,
        )

    def _call_mcp(self, tool: ToolConfig, query: str,
                  params: Dict = None) -> ToolResult:
        """Call an MCP server tool."""
        try:
            import urllib.request
            data = json.dumps({
                "jsonrpc": "2.0", "method": "tools/call",
                "params": {"name": tool.name, "arguments": {"query": query, **(params or {})}},
                "id": 1,
            }).encode()
            req = urllib.request.Request(
                tool.server_url, data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                response = json.loads(resp.read().decode())
            result_text = response.get("result", {})
            if isinstance(result_text, dict):
                content = result_text.get("content", [])
                if content and isinstance(content, list):
                    result_text = content[0].get("text", str(result_text))
                else:
                    result_text = json.dumps(result_text)
            return ToolResult(tool_name=tool.name, result=str(result_text), success=True)
        except Exception as e:
            return ToolResult(tool_name=tool.name, result="", success=False, error=str(e))

    def _call_api(self, tool: ToolConfig, query: str,
                  params: Dict = None) -> ToolResult:
        """Call a REST API tool."""
        try:
            import urllib.request
            data = json.dumps({"query": query, **(params or {})}).encode()
            headers = {"Content-Type": "application/json"}
            if tool.api_key:
                headers["Authorization"] = f"Bearer {tool.api_key}"
            req = urllib.request.Request(
                tool.api_endpoint, data=data, headers=headers,
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = resp.read().decode()
            return ToolResult(tool_name=tool.name, result=result, success=True)
        except Exception as e:
            return ToolResult(tool_name=tool.name, result="", success=False, error=str(e))

    def select_tools(self, query: str, query_type: str = "",
                     domain: str = "") -> List[str]:
        """Suggest which tools to call for a query (RL can override)."""
        selected = []
        query_lower = query.lower()

        for name, tool in self._tools.items():
            if not tool.enabled:
                continue
            # Simple keyword matching for tool selection
            desc_lower = tool.description.lower()
            name_lower = name.lower()
            if any(term in query_lower for term in name_lower.split()):
                selected.append(name)
            elif any(term in query_lower for term in desc_lower.split()[:5]):
                selected.append(name)

        return selected

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name, "type": t.tool_type,
                "description": t.description, "enabled": t.enabled,
                "calls": t.call_count, "success_rate": t.success_count / max(t.call_count, 1),
                "avg_latency": f"{t.avg_latency:.2f}s",
            }
            for t in self._tools.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_tools": len(self._tools),
            "total_calls": sum(t.call_count for t in self._tools.values()),
            "tools": self.list_tools(),
        }


class MCPServer:
    """Serve adaptive-intelligence as an MCP tool.

    Any MCP client (Claude, Cursor, custom apps) can connect
    and use the retrieval as a tool.
    """

    def __init__(self, engine):
        self.engine = engine
        self._running = False

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return MCP tool definition."""
        return {
            "name": "adaptive_intelligence_search",
            "description": "Search documents using adaptive RL-based retrieval. "
                          "Learns which strategy works best per query type.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["text", "json", "csv"],
                        "description": "Output format",
                        "default": "text",
                    },
                },
                "required": ["query"],
            },
        }

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        method = request.get("method", "")

        if method == "tools/list":
            return {
                "tools": [self.get_tool_definition()],
            }

        elif method == "tools/call":
            params = request.get("params", {})
            args = params.get("arguments", {})
            query = args.get("query", "")
            output_format = args.get("output_format", "text")

            response = self.engine.ask(query, output_format=output_format if output_format != "text" else None)

            return {
                "content": [{
                    "type": "text",
                    "text": response.answer,
                }],
                "metadata": {
                    "confidence": response.confidence,
                    "strategy": response.retrieval_strategy,
                    "citations": [
                        {"source": c.source_document, "page": c.page}
                        for c in (response.citations or [])[:5]
                    ],
                },
            }

        return {"error": f"Unknown method: {method}"}

    def serve(self, port: int = 8080):
        """Start MCP server (HTTP/SSE)."""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            server_ref = self

            class MCPHandler(BaseHTTPRequestHandler):
                def do_POST(self):
                    length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(length).decode()
                    request = json.loads(body)

                    response = server_ref.handle_request(request)

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "jsonrpc": "2.0",
                        "result": response,
                        "id": request.get("id", 1),
                    }).encode())

                def log_message(self, format, *args):
                    logger.debug(format % args)

            server = HTTPServer(('0.0.0.0', port), MCPHandler)
            self._running = True
            logger.info(f"MCP server started on port {port}")
            print(f"MCP server running on http://localhost:{port}")
            server.serve_forever()

        except KeyboardInterrupt:
            self._running = False
            logger.info("MCP server stopped")
