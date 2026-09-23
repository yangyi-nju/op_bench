"""MCP 2025-11-25 tool endpoint; local transport avoids a second tool service.

The fixed harness exchanges serialized JSON-RPC with this endpoint.
Only the tools capability is implemented: no resources, prompts or sampling.
"""
from __future__ import annotations

import json

from op_bench.runner.tools import TOOL_VERSION


PROTOCOL_VERSION = "2025-11-25"


class MCPServer:
    def __init__(self, executor):
        self.executor = executor
        self.initialized = False

    def exchange(self, message: str, timeout_sec: float = 120) -> str | None:
        identifier = None
        try:
            request = json.loads(message)
            if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                raise ValueError("invalid JSON-RPC request")
            identifier = request.get("id")
            method, params = request.get("method"), request.get("params", {})
            if method == "notifications/initialized":
                return None
            if identifier is None:
                raise ValueError("request id required")
            if method == "initialize":
                self.initialized = True
                result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                          "serverInfo": {"name": "opbench-tools", "version": TOOL_VERSION}}
            elif method == "ping":
                result = {}
            elif not self.initialized:
                raise ValueError("initialize first")
            elif method == "tools/list":
                result = {"tools": [{"name": item["function"]["name"],
                    "description": item["function"]["description"],
                    "inputSchema": item["function"]["parameters"]}
                    for item in self.executor.definitions]}
            elif method == "tools/call":
                value = self.executor.call(params["name"], params.get("arguments", {}), timeout_sec)
                result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
                          "structuredContent": value, "isError": bool(value.get("is_error"))}
            else:
                return json.dumps({"jsonrpc": "2.0", "id": identifier,
                                   "error": {"code": -32601, "message": "method not supported"}})
            return json.dumps({"jsonrpc": "2.0", "id": identifier, "result": result}, ensure_ascii=False)
        except (ValueError, TypeError, KeyError) as exc:
            return json.dumps({"jsonrpc": "2.0", "id": identifier,
                               "error": {"code": -32602, "message": str(exc)}})

class MCPToolClient:
    """The built-in harness uses a local serialized JSON-RPC transport."""
    def __init__(self, executor):
        self.server = MCPServer(executor)
        self.sequence = 0
        self._request("initialize", {"protocolVersion": PROTOCOL_VERSION,
            "capabilities": {}, "clientInfo": {"name": "opbench", "version": "0.8"}}, 10)
        self.server.exchange(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        result = self._request("tools/list", {}, 10)
        self.definitions = [{"type": "function", "function": {"name": tool["name"],
                            "description": tool["description"], "parameters": tool["inputSchema"]}}
                            for tool in result["tools"]]

    @property
    def finished(self):
        return self.server.executor.finished

    def _request(self, method, params, timeout_sec):
        self.sequence += 1
        reply = json.loads(self.server.exchange(json.dumps({"jsonrpc": "2.0", "id": self.sequence,
                            "method": method, "params": params}), timeout_sec))
        if reply.get("id") != self.sequence or "error" in reply:
            raise ValueError("MCP request failed: " + str(reply.get("error")))
        return reply["result"]

    def call(self, name, arguments, timeout_sec):
        return self._request("tools/call", {"name": name, "arguments": arguments}, timeout_sec)["structuredContent"]
