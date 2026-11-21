import asyncio
from mcp_client import run_mcp_session


def test_mcp_client_readonly_permissions():
    out = asyncio.run(run_mcp_session("tok-readonly-ops"))
    assert set(out["available_tools"]) == {"crm_get_customer", "ticket_get_status", "ticket_create"}
    assert out["crm_result"]["authorized"] is True
    assert out["crm_result"]["customer"]["name"] == "Tariq Al-Mansoor"
    assert out["ticket_create_result"]["authorized"] is False
    assert "Forbidden" in out["ticket_create_result"]["error"]


def test_mcp_client_compliance_lead_permissions():
    out = asyncio.run(run_mcp_session("tok-compliance-lead"))
    assert out["crm_result"]["authorized"] is True
    assert out["ticket_create_result"]["authorized"] is True
    assert out["ticket_create_result"]["created"] is True


def test_mcp_client_invalid_token_blocked():
    out = asyncio.run(run_mcp_session("invalid-token"))
    assert out["crm_result"]["authorized"] is False
    assert "Unauthorized" in out["crm_result"]["error"]