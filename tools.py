import time
from typing import Literal, cast
from pydantic import BaseModel, Field, ValidationError


# ==========================================
# 1. PYDANTIC SCHEMAS (INPUT & OUTPUT CONTRACTS)
# ==========================================

class DatabaseQueryInput(BaseModel):
    account_id: str = Field(..., min_length=3, max_length=25, description="Account or customer ID, e.g. ACC-102")
    simulate_timeout: bool = False

class DatabaseQueryOutput(BaseModel):
    found: bool
    account_id: str
    holder_name: str | None = None
    status: Literal["active", "under_review", "restricted", "timeout", "error"] = "error"
    balance_eur: float = 0.0
    message: str


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
    requires_human_approval: bool = True


class TicketCreationInput(BaseModel):
    title: str = Field(..., min_length=3, max_length=150)
    priority: Literal["low", "medium", "high"]
    description: str = Field(..., min_length=5, max_length=1500)
    account_id: str | None = None

class TicketCreationOutput(BaseModel):
    ticket_id: str
    status: Literal["pending_approval"]
    priority: str
    title: str
    requires_human_approval: bool = True


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


# ==========================================
# 2. LOCAL SAMPLE DATA (FOR PHASE 1 TESTING)
# ==========================================

MOCK_ACCOUNTS = {
    "ACC-101": {"holder_name": "Clara Weber", "status": "active", "balance_eur": 8400.0},
    "ACC-102": {"holder_name": "Tariq Al-Mansoor", "status": "under_review", "balance_eur": 25000.0},
}

MOCK_DOCS = [
    ("compliance", "Transfers above EUR 10,000 require secondary compliance review before settlement.", "Payment_Controls_Policy.md"),
    ("hr", "Employees must submit leave requests with dates and category.", "Employee_Handbook.md"),
    ("operations", "High-priority tickets require human approval before closure or escalation.", "Ops_Runbook.md"),
]

MOCK_TICKETS = {
    "TKT-101": {"state": "pending_approval", "owner": "compliance_reviewer"},
    "TKT-102": {"state": "open", "owner": "ops_analyst"},
}

MOCK_AUDIT_LOGS = {
    "ACC-102": ["account_queried", "policy_searched"],
}


# ==========================================
# 3. THE 8 TOOLS (WITH VALIDATION & TIMEOUTS)
# ==========================================

def database_query(raw_args: dict, timeout_seconds: float = 2.0) -> dict:
    start = time.time()
    try:
        args = DatabaseQueryInput(**raw_args)
        if args.simulate_timeout:
            time.sleep(0.05)
            raise TimeoutError("Database query timed out")
        if (time.time() - start) > timeout_seconds:
            raise TimeoutError("Database query exceeded timeout budget")

        key = args.account_id.upper()
        rec = MOCK_ACCOUNTS.get(key)
        if not rec:
            return DatabaseQueryOutput(
                found=False,
                account_id=key,
                status="error",
                message="Account record not found",
            ).model_dump()

        return DatabaseQueryOutput(
            found=True,
            account_id=key,
            holder_name=rec["holder_name"],
            status=rec["status"],
            balance_eur=rec["balance_eur"],
            message="Record retrieved",
        ).model_dump()
    except ValidationError as exc:
        return {"error": f"Invalid input for database_query: {exc.errors()[0]['msg']}"}
    except TimeoutError as exc:
        return DatabaseQueryOutput(
            found=False,
            account_id=str(raw_args.get("account_id", "UNKNOWN")),
            status="timeout",
            message=str(exc),
        ).model_dump()


def document_search(raw_args: dict) -> dict:
    try:
        args = DocumentSearchInput(**raw_args)
        q = args.query.lower().strip()
        matches: list[str] = []
        sources: list[str] = []
        for cat, text, src in MOCK_DOCS:
            cat_ok = args.category == "all" or args.category == cat
            if cat_ok and (q in text.lower() or any(word in text.lower() for word in q.split() if len(word) > 3)):
                matches.append(text)
                sources.append(src)
        return DocumentSearchOutput(
            query=args.query,
            category=args.category,
            matches=matches,
            sources=sources,
        ).model_dump()
    except ValidationError as exc:
        return {"error": f"Invalid input for document_search: {exc.errors()[0]['msg']}"}


def calendar_lookup(raw_args: dict) -> dict:
    try:
        args = CalendarLookupInput(**raw_args)
        slots = ["09:30-10:00", "14:00-14:30"] if "compliance" in args.calendar_name.lower() else ["11:00-11:30", "16:00-16:30"]
        return CalendarLookupOutput(
            calendar_name=args.calendar_name,
            date_iso=args.date_iso,
            available_slots=slots,
        ).model_dump()
    except ValidationError as exc:
        return {"error": f"Invalid input for calendar_lookup: {exc.errors()[0]['msg']}"}


def email_draft(raw_args: dict) -> dict:
    try:
        args = EmailDraftInput(**raw_args)
        return EmailDraftOutput(
            draft_id="DRAFT-1001",
            status="draft_created",
            to=args.to,
            subject=args.subject,
            requires_human_approval=True,
        ).model_dump()
    except ValidationError as exc:
        return {"error": f"Invalid input for email_draft: {exc.errors()[0]['msg']}"}


def ticket_creation(raw_args: dict) -> dict:
    try:
        args = TicketCreationInput(**raw_args)
        return TicketCreationOutput(
            ticket_id="TKT-2001",
            status="pending_approval",
            priority=args.priority,
            title=args.title,
            requires_human_approval=True,
        ).model_dump()
    except ValidationError as exc:
        return {"error": f"Invalid input for ticket_creation: {exc.errors()[0]['msg']}"}


def status_lookup(raw_args: dict) -> dict:
    try:
        args = StatusLookupInput(**raw_args)
        key = args.ticket_id.upper()
        rec = MOCK_TICKETS.get(key)
        if not rec:
            return StatusLookupOutput(found=False, ticket_id=key, state="not_found").model_dump()
        state_val = cast(
            Literal["open", "pending_approval", "resolved", "not_found"],
            rec["state"],
        )
        return StatusLookupOutput(
            found=True,
            ticket_id=key,
            state=state_val,
            owner=rec["owner"],
        ).model_dump()
    except ValidationError as exc:
        return {"error": f"Invalid input for status_lookup: {exc.errors()[0]['msg']}"}


def threshold_check(raw_args: dict) -> dict:
    try:
        args = ThresholdCheckInput(**raw_args)
        diff = round(args.amount - args.threshold, 2)
        return ThresholdCheckOutput(
            label=args.label,
            amount=args.amount,
            threshold=args.threshold,
            exceeds_threshold=args.amount > args.threshold,
            difference=diff,
        ).model_dump()
    except ValidationError as exc:
        return {"error": f"Invalid input for threshold_check: {exc.errors()[0]['msg']}"}


def audit_log_lookup(raw_args: dict) -> dict:
    try:
        args = AuditLogLookupInput(**raw_args)
        key = args.entity_id.upper()
        return AuditLogLookupOutput(
            entity_id=key,
            events=MOCK_AUDIT_LOGS.get(key, []),
        ).model_dump()
    except ValidationError as exc:
        return {"error": f"Invalid input for audit_log_lookup: {exc.errors()[0]['msg']}"}


TOOLS = {
    "database_query": database_query,
    "document_search": document_search,
    "calendar_lookup": calendar_lookup,
    "email_draft": email_draft,
    "ticket_creation": ticket_creation,
    "status_lookup": status_lookup,
    "threshold_check": threshold_check,
    "audit_log_lookup": audit_log_lookup,
}