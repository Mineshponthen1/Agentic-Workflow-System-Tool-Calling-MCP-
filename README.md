# Agentic Workflow System — Tool Calling, MCP & LangGraph

Local Python workflow combining **Ollama (`llama3.2`)** tool calling, **Pydantic** validation, **Model Context Protocol (`mcp`)** over `stdio` with role-based access control, **LangGraph** state orchestration with approval gates, `pytest` tests, and failure-taxonomy evaluations.

## Architecture

```text
Workflow Goal
      │
      ▼
LangGraph StateGraph (orchestrator.py)
  plan ──► approval_gate ──► act_and_observe
      │
      ├── Direct Tool Loop (tools.py, agent_runner.py)
      │     Pydantic I/O schemas, timeouts, safe error fallbacks
      │
      └── MCP Server over stdio (mcp_server.py, mcp_client.py)
            Token auth, role allow-lists, per-tool RBAC

Phases
Phase 1 — Tool-Calling Foundation (tools.py, agent_runner.py, test_phase1.py)
Validated Tools: 8 Pydantic-schemed tools (database_query, document_search, calendar_lookup, email_draft, ticket_creation, status_lookup, threshold_check, audit_log_lookup) with input and output validation.

Agent Loop: Sends prompts and tool schemas to llama3.2, executes tool calls within a step cap, and returns structured results.

Failure Handling: Catches validation errors, handles timeouts (such as database_query), and rejects unknown or malformed calls safely.

Tests: 5 pytest unit tests in test_phase1.py.

Phase 2 — MCP Server & RBAC (mcp_server.py, mcp_client.py, test_phase2.py)
MCP Server: Exposes mock CRM (crm_get_customer) and ticketing (ticket_get_status, ticket_create) tools over stdio using the mcp SDK.

Auth & Permissions: Authenticates caller tokens (tok-readonly-ops, tok-compliance-lead), rejects invalid tokens (Unauthorized), and enforces role allow-lists (ops_analyst read-only with Forbidden on ticket_create; compliance_lead allowed to create tickets).

MCP Client: Connects over stdio, initializes the session, lists tools dynamically, and tests role permissions.

Tests: 3 pytest unit tests in test_phase2.py.

Phase 3 — LangGraph Orchestration & Evaluation (orchestrator.py, eval_and_trace.py, test_phase3.py)
State Graph: Uses LangGraph (plan -> approval_gate -> act_and_observe) to track goal, step_count, max_steps, pending_tool, approved, observations, status, and summary.

Approval Gate: Pauses (awaiting_approval) before state-changing tools (ticket_creation, email_draft) until resumed with approve_pending=True.

Step Budget: Enforces max_steps and stops with max_steps_reached when exceeded.

Evaluation & Tracing: Classifies scripted runs into none, validation_error, timeout_error, permission_denied, budget_exceeded, and unexpected_outcome, with optional Langfuse tracing when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

Tests: 3 pytest unit tests in test_phase3.py plus 4 scripted scenarios in eval_and_trace.py.