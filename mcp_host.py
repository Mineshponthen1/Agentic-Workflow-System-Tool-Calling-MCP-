"""The agent host's MCP client: one connection to BankingInternalMCP per caller.

The host:
  - starts the server over stdio with the caller's token in its environment,
  - turns the server's tool list into the tool definitions the model sees
    (so there is one source of truth for tool schemas: the server),
  - reads each tool's read_only_hint to decide which calls need human approval,
  - calls tools with a real timeout and classifies every failure.
"""

import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = str(Path(__file__).resolve().parent / "mcp_server.py")


@dataclass
class ToolOutcome:
    ok: bool
    result: dict[str, Any]
    error_type: str | None = None  # validation_error | permission_denied | timeout | tool_error


@dataclass
class MCPToolHost:
    token: str
    timeout_seconds: float = 3.0
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    read_only: dict[str, bool] = field(default_factory=dict)
    _stack: AsyncExitStack | None = None
    _session: ClientSession | None = None

    async def __aenter__(self) -> "MCPToolHost":
        env = {**os.environ, "MCP_CALLER_TOKEN": self.token}
        params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT], env=env)
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        await self.refresh_tools()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack:
            await self._stack.aclose()

    async def refresh_tools(self) -> None:
        assert self._session
        listed = await self._session.list_tools()
        self.tool_definitions = []
        self.read_only = {}
        for tool in listed.tools:
            self.tool_definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": simplify_schema(tool.input_schema),
                },
            })
            hint = tool.annotations.read_only_hint if tool.annotations else None
            self.read_only[tool.name] = bool(hint)

    @property
    def tool_names(self) -> set[str]:
        return set(self.read_only)

    def needs_approval(self, tool_name: str) -> bool:
        """Anything the server does not declare read-only is treated as state-changing."""
        return not self.read_only.get(tool_name, False)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> ToolOutcome:
        assert self._session
        try:
            res = await self._session.call_tool(tool_name, arguments, read_timeout_seconds=self.timeout_seconds)
        except Exception as exc:  # the MCP client raises on a read timeout
            text = str(exc) or type(exc).__name__
            if "timed out" in text.lower() or "timeout" in text.lower() or "TimeoutError" in type(exc).__name__:
                return ToolOutcome(False, {"error": f"'{tool_name}' timed out after {self.timeout_seconds:g}s"}, "timeout")
            return ToolOutcome(False, {"error": text}, "tool_error")

        text = "".join(getattr(c, "text", "") for c in (getattr(res, "content", None) or []))
        if getattr(res, "is_error", False):
            return ToolOutcome(False, {"error": text}, classify_tool_error(text))
        structured = getattr(res, "structured_content", None)
        if isinstance(structured, dict):
            result = structured.get("result", structured) if set(structured) == {"result"} else structured
        else:
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                result = {"text": text}
        return ToolOutcome(True, result)


def simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrites an MCP (Pydantic) JSON schema into the plain form local models handle well.

    Optional fields arrive as anyOf [X, null], which the Ollama client drops, leaving the
    field with no type. This keeps X, and removes titles and null defaults (noise for the model).
    """
    props = {}
    for name, prop in (schema.get("properties") or {}).items():
        prop = dict(prop)
        options = [o for o in prop.pop("anyOf", []) if o.get("type") != "null"]
        if len(options) == 1:
            prop = {**options[0], **prop}
        prop.pop("title", None)
        if prop.get("default", "x") is None:
            prop.pop("default")
        props[name] = prop
    out = {"type": "object", "properties": props}
    if schema.get("required"):
        out["required"] = list(schema["required"])
    return out


def classify_tool_error(message: str) -> str:
    low = message.lower()
    if "forbidden" in low or "unauthorized" in low:
        return "permission_denied"
    if "invalid input" in low or "validation" in low or "field required" in low or "input should" in low:
        return "validation_error"
    if "unknown tool" in low or "not found" in low and "tool" in low:
        return "unknown_tool"
    return "tool_error"
