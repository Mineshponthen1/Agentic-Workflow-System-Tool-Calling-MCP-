"""The MCP server and host: auth at the connection, allow-lists, permission checks, real timeouts."""

import asyncio
import time

from mcp_host import MCPToolHost

READ_ONLY = {"audit_log_lookup", "calendar_lookup", "crm_get_customer", "database_query",
             "document_search", "status_lookup", "threshold_check"}


def run(coro):
    return asyncio.run(coro)


def test_ops_analyst_only_sees_read_only_tools():
    async def go():
        async with MCPToolHost("tok-readonly-ops") as host:
            return host.tool_names, {n: host.needs_approval(n) for n in host.tool_names}
    names, approval = run(go())
    assert names == READ_ONLY
    assert not any(approval.values())


def test_compliance_lead_sees_all_and_write_tools_need_approval():
    async def go():
        async with MCPToolHost("tok-compliance-lead") as host:
            return host.tool_names, {n for n in host.tool_names if host.needs_approval(n)}
    names, gated = run(go())
    assert len(names) == 9
    assert gated == {"email_draft", "ticket_creation"}


def test_forbidden_tool_is_refused_even_if_called_directly():
    async def go():
        async with MCPToolHost("tok-readonly-ops") as host:
            return await host.call("ticket_creation", {"title": "x", "priority": "high", "description": "hello"})
    out = run(go())
    assert not out.ok and out.error_type == "permission_denied"
    assert "Forbidden" in out.result["error"]


def test_invalid_token_sees_nothing_and_is_refused():
    async def go():
        async with MCPToolHost("not-a-real-token") as host:
            return host.tool_names, await host.call("database_query", {"account_id": "ACC-101"})
    names, out = run(go())
    assert names == set()
    assert out.error_type == "permission_denied" and "Unauthorized" in out.result["error"]


def test_bad_arguments_are_a_validation_error():
    async def go():
        async with MCPToolHost("tok-readonly-ops") as host:
            return await host.call("threshold_check", {"amount": "lots", "threshold": 5})
    out = run(go())
    assert out.error_type == "validation_error"


def test_slow_tool_times_out_and_session_keeps_working():
    async def go():
        async with MCPToolHost("tok-readonly-ops", timeout_seconds=1.0) as host:
            start = time.time()
            slow = await host.call("database_query", {"account_id": "ACC-999"})
            took = time.time() - start
            after = await host.call("database_query", {"account_id": "ACC-101"})
            return slow, took, after
    slow, took, after = run(go())
    assert slow.error_type == "timeout" and took < 3
    assert after.ok and after.result["holder_name"] == "Clara Weber"


def test_tool_schemas_reach_ollama_with_types_intact():
    from ollama._types import ChatRequest

    async def go():
        async with MCPToolHost("tok-compliance-lead") as host:
            return host.tool_definitions
    defs = run(go())
    req = ChatRequest(model="llama3.2", messages=[{"role": "user", "content": "hi"}], tools=defs).model_dump(exclude_none=True)
    for tool in req["tools"]:
        for name, prop in tool["function"]["parameters"]["properties"].items():
            assert "type" in prop, f"{tool['function']['name']}.{name} lost its type"
    ticket = next(t for t in req["tools"] if t["function"]["name"] == "ticket_creation")
    assert ticket["function"]["parameters"]["properties"]["account_id"]["type"] == "string"
    assert ticket["function"]["parameters"]["required"] == ["title", "priority", "description"]
