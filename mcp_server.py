import json
from typing import Literal
from mcp.server import MCPServer


mcp = MCPServer("BankingInternalMCP")

# Caller tokens mapped to roles and explicit tool allow-lists
CALLER_PERMISSIONS: dict[str, dict[str, object]] = {
    "tok-readonly-ops": {
        "role": "ops_analyst",
        "allowed_tools": {"crm_get_customer", "ticket_get_status"},
    },
    "tok-compliance-lead": {
        "role": "compliance_lead",
        "allowed_tools": {"crm_get_customer", "ticket_get_status", "ticket_create"},
    },
}

# Mock CRM store
MOCK_CRM_CUSTOMERS: dict[str, dict[str, str]] = {
    "ACC-101": {"name": "Clara Weber", "tier": "standard", "kyc_status": "verified"},
    "ACC-102": {"name": "Tariq Al-Mansoor", "tier": "priority", "kyc_status": "pending_review"},
}

# Mock Ticketing store
MOCK_TICKETS_DB: dict[str, dict[str, str]] = {
    "TKT-101": {
        "account_id": "ACC-102",
        "status": "open",
        "priority": "high",
        "title": "KYC refresh check",
    },
}

_ticket_counter = 102


def authorize_tool_call(token: str, tool_name: str) -> tuple[bool, str]:
    """Checks caller authentication token, role allow-list, and per-tool permission."""
    caller = CALLER_PERMISSIONS.get(token)
    if not caller:
        return False, "Unauthorized: invalid or missing token."

    allowed = caller["allowed_tools"]
    assert isinstance(allowed, set)
    if tool_name not in allowed:
        return False, f"Forbidden: role '{caller['role']}' is not allowed to call '{tool_name}'."

    return True, "Authorized"


@mcp.tool()
def crm_get_customer(token: str, account_id: str) -> str:
    """Look up a customer profile from the mock CRM (requires read permission)."""
    ok, reason = authorize_tool_call(token, "crm_get_customer")
    if not ok:
        return json.dumps({"authorized": False, "error": reason})

    key = account_id.strip().upper()
    customer = MOCK_CRM_CUSTOMERS.get(key)
    if not customer:
        return json.dumps({"authorized": True, "found": False, "account_id": key})

    return json.dumps({
        "authorized": True,
        "found": True,
        "account_id": key,
        "customer": customer,
    })


@mcp.tool()
def ticket_get_status(token: str, ticket_id: str) -> str:
    """Check a ticket's status in the mock ticketing system (requires read permission)."""
    ok, reason = authorize_tool_call(token, "ticket_get_status")
    if not ok:
        return json.dumps({"authorized": False, "error": reason})

    key = ticket_id.strip().upper()
    ticket = MOCK_TICKETS_DB.get(key)
    if not ticket:
        return json.dumps({"authorized": True, "found": False, "ticket_id": key})

    return json.dumps({
        "authorized": True,
        "found": True,
        "ticket_id": key,
        "ticket": ticket,
    })


@mcp.tool()
def ticket_create(
    token: str,
    account_id: str,
    title: str,
    priority: Literal["low", "medium", "high"],
) -> str:
    """Create a ticket in the mock ticketing system (requires ticket_create permission)."""
    global _ticket_counter
    ok, reason = authorize_tool_call(token, "ticket_create")
    if not ok:
        return json.dumps({"authorized": False, "error": reason})

    _ticket_counter += 1
    new_id = f"TKT-{_ticket_counter}"
    MOCK_TICKETS_DB[new_id] = {
        "account_id": account_id.strip().upper(),
        "status": "pending_approval",
        "priority": priority,
        "title": title.strip(),
    }
    return json.dumps({
        "authorized": True,
        "created": True,
        "ticket_id": new_id,
        "ticket": MOCK_TICKETS_DB[new_id],
    })


if __name__ == "__main__":
    mcp.run()