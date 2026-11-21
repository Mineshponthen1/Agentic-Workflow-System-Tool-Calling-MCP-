import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_SCRIPT = str(Path(__file__).resolve().parent / "mcp_server.py")


async def run_mcp_session(token: str) -> dict[str, Any]:
    """Connects to mcp_server.py over stdio MCP, lists tools, and exercises CRM + ticketing."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            available_tools = [t.name for t in tools_response.tools]

            crm_res = await session.call_tool(
                "crm_get_customer",
                arguments={"token": token, "account_id": "ACC-102"},
            )
            crm_text = getattr(crm_res.content[0], "text", "{}") if crm_res.content else "{}"

            ticket_res = await session.call_tool(
                "ticket_create",
                arguments={
                    "token": token,
                    "account_id": "ACC-102",
                    "title": "High-value transfer review",
                    "priority": "high",
                },
            )
            ticket_text = getattr(ticket_res.content[0], "text", "{}") if ticket_res.content else "{}"

            return {
                "available_tools": available_tools,
                "crm_result": json.loads(crm_text),
                "ticket_create_result": json.loads(ticket_text),
            }


def main() -> None:
    print("--- 1. Read-Only Ops Token over MCP ---")
    ops_out = asyncio.run(run_mcp_session("tok-readonly-ops"))
    print(json.dumps(ops_out, indent=2))

    print("\n--- 2. Compliance Lead Token over MCP ---")
    lead_out = asyncio.run(run_mcp_session("tok-compliance-lead"))
    print(json.dumps(lead_out, indent=2))


if __name__ == "__main__":
    main()