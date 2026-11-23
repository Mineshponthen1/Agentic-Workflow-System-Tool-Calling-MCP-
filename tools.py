"""Business logic for the 9 banking tools.

Every tool takes a plain dict of arguments, validates it with a Pydantic input
model, and returns a dict validated by a Pydantic output model. The MCP server
(mcp_server.py) is the only thing that exposes these tools to an agent.
"""

from typing import Any, Literal, cast

from pydantic import BaseModel, Field, ValidationError


# ==========================================
# 1. PYDANTIC SCHEMAS (INPUT & OUTPUT CONTRACTS)
# ==========================================

class DatabaseQueryInput(BaseModel):
    account_id: str = Field(..., min_length=3, max_length=25, description="Account ID, e.g. ACC-102")

class DatabaseQueryOutput(BaseModel):
    found: bool
    account_id: str
    holder_name: str | None = None
    status: Literal["active", "under_review", "restricted", "not_found"]
    balance_eur: float | None = None


class DocumentSearchInput(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    category: Literal["all", "compliance", "hr", "operations"] = "all"

class DocumentSearchOutput(BaseModel):
    query: str
    category: str
    matches: list[str]
    sources: list[str]


class CalendarLookupInput(BaseModel):
    calendar_name: str = Field(..., min_length=2, max_length=50)
    date_iso: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Format YYYY-MM-DD")

class CalendarLookupOutput(BaseModel):
    calendar_name: str
    date_iso: str
    available_slots: list[str]


class EmailDraftInput(BaseModel):
    to: str = Field(..., min_length=5, max_length=150)
    subject: str = Field(..., min_length=3, max_length=160)
    body: str = Field(..., min_length=5, max_length=2500)

class EmailDraftOutput(BaseModel):
    draft_id: str
    status: Literal["draft_created"]
    to: str
    subject: str


class TicketCreationInput(BaseModel):
    title: str = Field(..., min_length=3, max_length=150)
    priority: Literal["low", "medium", "high"]
    description: str = Field(..., min_length=5, max_length=1500)
    account_id: str | None = None

class TicketCreationOutput(BaseModel):
    ticket_id: str
    status: Literal["open"]
    priority: str
    title: str
    account_id: str | None = None


class StatusLookupInput(BaseModel):
    ticket_id: str = Field(..., min_length=3, max_length=30)

class StatusLookupOutput(BaseModel):
    found: bool
    ticket_id: str
    state: Literal["open", "pending_approval", "resolved", "not_found"]
    owner: str | None = None


class ThresholdCheckInput(BaseModel):
    amount: float = Field(..., ge=0)
    threshold: float = Field(..., gt=0)
    label: str = Field(default="amount_check", min_length=1, max_length=80)

class ThresholdCheckOutput(BaseModel):
    label: str
    amount: float
    threshold: float
    exceeds_threshold: bool
    difference: float


class AuditLogLookupInput(BaseModel):
    entity_id: str = Field(..., min_length=2, max_length=40)

class AuditLogLookupOutput(BaseModel):
    entity_id: str
    events: list[str]


class CrmGetCustomerInput(BaseModel):
    account_id: str = Field(..., min_length=3, max_length=25)

class CrmGetCustomerOutput(BaseModel):
    found: bool
    account_id: str
    name: str | None = None
    tier: Literal["standard", "priority"] | None = None
    kyc_status: Literal["verified", "pending_review", "expired"] | None = None


# ==========================================
# 2. MOCK BACK-END DATA
# ==========================================

MOCK_ACCOUNTS = {
    "ACC-101": {"holder_name": "Clara Weber", "status": "active", "balance_eur": 8400.0},
    "ACC-102": {"holder_name": "Tariq Al-Mansoor", "status": "under_review", "balance_eur": 25000.0},
    "ACC-103": {"holder_name": "Lena Fischer", "status": "restricted", "balance_eur": 120500.0},
    "ACC-999": {"holder_name": "Legacy Test Holder", "status": "active", "balance_eur": 500.0},
}

# Accounts whose core-banking lookup is slow, in seconds. Used to exercise real
# timeouts: the agent host gives up long before ACC-999 answers.
SLOW_ACCOUNTS = {"ACC-999": 6.0}

MOCK_CRM = {
    "ACC-101": {"name": "Clara Weber", "tier": "standard", "kyc_status": "verified"},
    "ACC-102": {"name": "Tariq Al-Mansoor", "tier": "priority", "kyc_status": "pending_review"},
    "ACC-103": {"name": "Lena Fischer", "tier": "priority", "kyc_status": "expired"},
}

MOCK_DOCS = [
    ("compliance", "Transfers above EUR 10,000 require secondary compliance review before settlement.", "Payment_Controls_Policy.md"),
    ("compliance", "Customers with expired KYC must not transact until KYC is refreshed.", "KYC_Policy.md"),
    ("hr", "Employees must submit leave requests with dates and category.", "Employee_Handbook.md"),
    ("operations", "High-priority tickets require human approval before closure or escalation.", "Ops_Runbook.md"),
]

MOCK_TICKETS = {
    "TKT-101": {"state": "pending_approval", "owner": "compliance_reviewer"},
    "TKT-102": {"state": "open", "owner": "ops_analyst"},
    "TKT-103": {"state": "resolved", "owner": "ops_analyst"},
}

MOCK_AUDIT_LOGS = {
    "ACC-102": ["2026-11-02 account_queried", "2026-11-03 policy_searched", "2026-11-05 review_opened"],
    "ACC-103": ["2026-10-20 kyc_expired", "2026-10-21 account_restricted"],
}

_ticket_counter = 2000
_draft_counter = 1000


# ==========================================
# 3. THE TOOLS
# ==========================================

def _invalid(tool: str, exc: ValidationError) -> dict[str, Any]:
    first = exc.errors()[0]
    field = ".".join(str(p) for p in first.get("loc", ())) or "arguments"
    return {"error": f"Invalid input for {tool}: {field}: {first['msg']}", "error_type": "validation_error"}


def database_query(raw_args: dict) -> dict:
    try:
        args = DatabaseQueryInput(**raw_args)
    except ValidationError as exc:
        return _invalid("database_query", exc)
    key = args.account_id.strip().upper()
    rec = MOCK_ACCOUNTS.get(key)
    if not rec:
        return DatabaseQueryOutput(found=False, account_id=key, status="not_found").model_dump()
    return DatabaseQueryOutput(found=True, account_id=key, **rec).model_dump()


def document_search(raw_args: dict) -> dict:
    try:
        args = DocumentSearchInput(**raw_args)
    except ValidationError as exc:
        return _invalid("document_search", exc)
    words = [w for w in args.query.lower().replace(",", " ").split() if len(w) > 3]
    matches: list[str] = []
    sources: list[str] = []
    for cat, text, src in MOCK_DOCS:
        if args.category not in ("all", cat):
            continue
        if args.query.lower() in text.lower() or any(w in text.lower() for w in words):
            matches.append(text)
            sources.append(src)
    return DocumentSearchOutput(query=args.query, category=args.category, matches=matches, sources=sources).model_dump()


def calendar_lookup(raw_args: dict) -> dict:
    try:
        args = CalendarLookupInput(**raw_args)
    except ValidationError as exc:
        return _invalid("calendar_lookup", exc)
    compliance = "compliance" in args.calendar_name.lower()
    slots = ["09:30-10:00", "14:00-14:30"] if compliance else ["11:00-11:30", "16:00-16:30"]
    return CalendarLookupOutput(calendar_name=args.calendar_name, date_iso=args.date_iso, available_slots=slots).model_dump()


def email_draft(raw_args: dict) -> dict:
    global _draft_counter
    try:
        args = EmailDraftInput(**raw_args)
    except ValidationError as exc:
        return _invalid("email_draft", exc)
    _draft_counter += 1
    return EmailDraftOutput(draft_id=f"DRAFT-{_draft_counter}", status="draft_created", to=args.to, subject=args.subject).model_dump()


def ticket_creation(raw_args: dict) -> dict:
    global _ticket_counter
    try:
        args = TicketCreationInput(**raw_args)
    except ValidationError as exc:
        return _invalid("ticket_creation", exc)
    _ticket_counter += 1
    ticket_id = f"TKT-{_ticket_counter}"
    MOCK_TICKETS[ticket_id] = {"state": "open", "owner": "compliance_reviewer"}
    return TicketCreationOutput(
        ticket_id=ticket_id, status="open", priority=args.priority, title=args.title,
        account_id=args.account_id.upper() if args.account_id else None,
    ).model_dump()


def status_lookup(raw_args: dict) -> dict:
    try:
        args = StatusLookupInput(**raw_args)
    except ValidationError as exc:
        return _invalid("status_lookup", exc)
    key = args.ticket_id.strip().upper()
    rec = MOCK_TICKETS.get(key)
    if not rec:
        return StatusLookupOutput(found=False, ticket_id=key, state="not_found").model_dump()
    state = cast(Literal["open", "pending_approval", "resolved", "not_found"], rec["state"])
    return StatusLookupOutput(found=True, ticket_id=key, state=state, owner=rec["owner"]).model_dump()


def threshold_check(raw_args: dict) -> dict:
    try:
        args = ThresholdCheckInput(**raw_args)
    except ValidationError as exc:
        return _invalid("threshold_check", exc)
    return ThresholdCheckOutput(
        label=args.label, amount=args.amount, threshold=args.threshold,
        exceeds_threshold=args.amount > args.threshold,
        difference=round(args.amount - args.threshold, 2),
    ).model_dump()


def audit_log_lookup(raw_args: dict) -> dict:
    try:
        args = AuditLogLookupInput(**raw_args)
    except ValidationError as exc:
        return _invalid("audit_log_lookup", exc)
    key = args.entity_id.strip().upper()
    return AuditLogLookupOutput(entity_id=key, events=MOCK_AUDIT_LOGS.get(key, [])).model_dump()


def crm_get_customer(raw_args: dict) -> dict:
    try:
        args = CrmGetCustomerInput(**raw_args)
    except ValidationError as exc:
        return _invalid("crm_get_customer", exc)
    key = args.account_id.strip().upper()
    rec = MOCK_CRM.get(key)
    if not rec:
        return CrmGetCustomerOutput(found=False, account_id=key).model_dump()
    return CrmGetCustomerOutput(found=True, account_id=key, **rec).model_dump()


TOOLS = {
    "database_query": database_query,
    "document_search": document_search,
    "calendar_lookup": calendar_lookup,
    "email_draft": email_draft,
    "ticket_creation": ticket_creation,
    "status_lookup": status_lookup,
    "threshold_check": threshold_check,
    "audit_log_lookup": audit_log_lookup,
    "crm_get_customer": crm_get_customer,
}

# Tools that change state in a back-end system. The MCP server marks these
# read_only_hint=False, and the agent host pauses for human approval before them.
STATE_CHANGING_TOOLS = {"email_draft", "ticket_creation"}
