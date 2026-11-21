# Agentic Workflow System — Tool Calling, MCP & LangGraph

A modular local AI workflow system in Python combining **Ollama (`llama3.2`)** tool calling, **Pydantic** schema contracts, **Model Context Protocol (`mcp`)** servers over `stdio` with role-based access control, **LangGraph** state orchestration with human-in-the-loop approval gates, **`pytest`** unit tests, and failure-taxonomy evaluations.

---

## Architecture

```text
                        ┌──────────────────────┐
                        │    Workflow Goal     │
                        └──────────┬───────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────────────┐
                 │  LangGraph StateGraph (orchestrator.py) │
                 │  plan ─► approval_gate ─► act_and_observe │
                 └─────────┬─────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│   Direct Tool Loop          │     │   MCP Server over stdio     │
│   tools.py + agent_runner.py│     │   mcp_server.py + mcp_client.py │
│   • Pydantic I/O schemas    │     │   • Caller token auth       │
│   • Timeout & error guards  │     │   • Role allow-lists (RBAC) │
└─────────────────────────────┘     └─────────────────────────────┘
Project Phases
Phase 1 — Tool-Calling Foundation (tools.py, agent_runner.py, test_phase1.py)
Establishes strict tool contracts and fault-tolerant execution between local Ollama and Python functions:

8 Pydantic-Validated Tools (tools.py): database_query, document_search, calendar_lookup, email_draft, ticket_creation, status_lookup, threshold_check, and audit_log_lookup—each enforcing input validation before execution and structured output schemas after execution.

Bounded Agent Loop (agent_runner.py): Sends prompts and tool definitions to llama3.2, executes requested tools safely within a step cap, feeds observations back to the model, and returns a structured response.

Resilience & Failure Handling: Catches Pydantic validation errors cleanly, handles tool timeouts (such as simulated timeouts on database_query), and safely rejects unknown or malformed tool calls.

Tests (test_phase1.py): 5 pytest unit tests covering tool registration, valid/timeout execution, invalid arguments, unknown tools, and loop behavior.

Phase 2 — Model Context Protocol & RBAC (mcp_server.py, mcp_client.py, test_phase2.py)
Isolates business tools behind an MCP stdio server with caller authentication and role-based permissions:

MCP Server (mcp_server.py): Exposes mock CRM (crm_get_customer) and ticketing (ticket_get_status, ticket_create) tools via the mcp SDK.

Authentication & Role-Based Access Control: Validates caller tokens (tok-readonly-ops, tok-compliance-lead), rejects invalid/missing tokens (Unauthorized), and enforces per-role allow-lists (ops_analyst has read access and is blocked with Forbidden on ticket_create; compliance_lead can create tickets).

MCP Client (mcp_client.py): Spawns the server over stdio, initializes a client session, discovers tools dynamically (list_tools), and runs permission checks across roles.

Tests (test_phase2.py): 3 pytest tests verifying read-only role restrictions, compliance-lead write access, and invalid-token rejection.

Phase 3 — LangGraph Orchestration, Approval Gates & Evaluation (orchestrator.py, eval_and_trace.py, test_phase3.py)
Adds stateful graph control, human approval checkpoints, step limits, and structured failure evaluation:

Stateful Graph (orchestrator.py): Uses a LangGraph StateGraph (plan -> approval_gate -> act_and_observe) tracking goal, step_count, max_steps, pending_tool, approved, observations, status, and summary.

Human-in-the-Loop Approval Gate: Pauses execution (awaiting_approval) before running state-changing tools (ticket_creation, email_draft) until explicitly resumed with approve_pending=True.

Step-Budget Guardrail: Enforces max_steps and halts with max_steps_reached if the budget is exceeded.

Failure Taxonomy & Tracing (eval_and_trace.py): Runs scripted scenarios and classifies outcomes into none, validation_error, timeout_error, permission_denied, budget_exceeded, and unexpected_outcome, with optional Langfuse tracing when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

Tests (test_phase3.py & eval_and_trace.py): 3 pytest unit tests plus 4 scripted scenario evaluations.

Running Tests & Demos (PowerShell)
PowerShell
# Run all 11 unit tests across Phases 1–3
python -m pytest -v

# Live Ollama tool-calling run (requires Ollama running with llama3.2)
python agent_runner.py

# MCP client-server role permission demo
python mcp_client.py

# LangGraph approval pause and resume demo
python orchestrator.py

# Scripted scenario evaluations and failure taxonomy report
python eval_and_trace.py
Notes & Scope
CRM, ticketing, and account records use in-memory mock fixtures for deterministic local runs and testing.

Langfuse trace export is optional and only active when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables are configured.