"""BankingInternalMCP: the mock core-banking, CRM and ticketing APIs as MCP tools.

Auth happens at the connection, not inside tool arguments: the agent host starts
this server with the caller's token in the MCP_CALLER_TOKEN environment variable
(the standard way to hand credentials to a stdio MCP server). The model never
sees the token and cannot change it.

Two layers of permissioning, from config/callers.json:
  1. tools/list only shows the tools the caller's role may use (allow-list), and
  2. every tools/call checks the role again (defence in depth).

Tokens are stored as SHA-256 hashes, never in plain text.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

import tools as backend

CALLERS_FILE = Path(__file__).resolve().parent / "config" / "callers.json"

READ = ToolAnnotations(read_only_hint=True, destructive_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)


def resolve_caller(token: str | None, callers_file: Path = CALLERS_FILE) -> dict[str, Any] | None:
    """Returns the caller record for a token, or None if the token is unknown."""
    if not token:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    for caller in json.loads(callers_file.read_text())["callers"]:
        if caller["token_sha256"] == digest:
            return caller
    return None


class PermissionedMCPServer(MCPServer):
    """An MCPServer that lists only the tools the connected caller is allowed to use."""

    def __init__(self, *args: Any, caller: dict[str, Any] | None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.caller = caller

    async def list_tools(self):  # type: ignore[override]
        allowed = set(self.caller["allowed_tools"]) if self.caller else set()
        return [t for t in await super().list_tools() if t.name in allowed]


CALLER = resolve_caller(os.environ.get("MCP_CALLER_TOKEN"))
mcp = PermissionedMCPServer("BankingInternalMCP", caller=CALLER)


def _run(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Checks permission, runs the back-end tool, and turns its errors into MCP tool errors."""
    if CALLER is None:
        raise ToolError("Unauthorized: invalid or missing caller token.")
    if tool_name not in CALLER["allowed_tools"]:
        raise ToolError(f"Forbidden: role '{CALLER['role']}' may not call '{tool_name}'.")
    result = backend.TOOLS[tool_name]({k: v for k, v in args.items() if v is not None})
    if "error" in result:
        raise ToolError(result["error"])
    return result


@mcp.tool(annotations=READ)
async def database_query(
    account_id: Annotated[str, Field(description="Account ID, e.g. ACC-102")],
) -> dict:
    """Look up an account's holder, status and balance in core banking."""
    delay = backend.SLOW_ACCOUNTS.get(account_id.strip().upper())
    if delay:
        await asyncio.sleep(delay)  # a slow legacy record, to exercise real timeouts
    return _run("database_query", {"account_id": account_id})


@mcp.tool(annotations=READ)
def document_search(
    query: Annotated[str, Field(description="Words to search policy documents for")],
    category: Literal["all", "compliance", "hr", "operations"] = "all",
) -> dict:
    """Search internal policy and operations documents."""
    return _run("document_search", {"query": query, "category": category})


@mcp.tool(annotations=READ)
def calendar_lookup(
    calendar_name: Annotated[str, Field(description="Team calendar, e.g. compliance")],
    date_iso: Annotated[str, Field(description="Date as YYYY-MM-DD")],
) -> dict:
    """List free meeting slots in a team calendar on a date."""
    return _run("calendar_lookup", {"calendar_name": calendar_name, "date_iso": date_iso})


@mcp.tool(annotations=WRITE)
def email_draft(
    to: Annotated[str, Field(description="Recipient email address")],
    subject: str,
    body: str,
) -> dict:
    """Create an email draft in the shared mailbox (does not send it). Needs human approval."""
    return _run("email_draft", {"to": to, "subject": subject, "body": body})


@mcp.tool(annotations=WRITE)
def ticket_creation(
    title: str,
    priority: Literal["low", "medium", "high"],
    description: str,
    account_id: Annotated[str | None, Field(description="Related account ID, if any")] = None,
) -> dict:
    """Open a support or compliance ticket in the ticketing system. Needs human approval."""
    return _run("ticket_creation", {"title": title, "priority": priority, "description": description, "account_id": account_id})


@mcp.tool(annotations=READ)
def status_lookup(
    ticket_id: Annotated[str, Field(description="Ticket ID, e.g. TKT-101")],
) -> dict:
    """Check the state and owner of an existing ticket."""
    return _run("status_lookup", {"ticket_id": ticket_id})


@mcp.tool(annotations=READ)
def threshold_check(
    amount: Annotated[float, Field(description="The amount to check, in EUR")],
    threshold: Annotated[float, Field(description="The limit to compare against, in EUR")],
    label: str = "amount_check",
) -> dict:
    """Compare an amount against a limit in code (never do this arithmetic yourself)."""
    return _run("threshold_check", {"amount": amount, "threshold": threshold, "label": label})


@mcp.tool(annotations=READ)
def audit_log_lookup(
    entity_id: Annotated[str, Field(description="Account or ticket ID")],
) -> dict:
    """List recorded audit events for an account or ticket."""
    return _run("audit_log_lookup", {"entity_id": entity_id})


@mcp.tool(annotations=READ)
def crm_get_customer(
    account_id: Annotated[str, Field(description="Account ID, e.g. ACC-102")],
) -> dict:
    """Look up a customer's CRM profile: name, tier and KYC status."""
    return _run("crm_get_customer", {"account_id": account_id})


if __name__ == "__main__":
    mcp.run()
