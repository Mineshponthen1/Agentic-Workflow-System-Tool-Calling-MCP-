# Agentic Workflow System (Tool Calling + MCP + LangGraph)

A modular local AI agentic workflow built in Python with Pydantic validation, Ollama, Model Context Protocol (`mcp`), LangGraph state orchestration, `pytest` unit tests, and scenario evaluations with a failure taxonomy.

## Architecture & Phases

* **Phase 1 — Tool-Calling Foundation (`tools.py`, `agent_runner.py`, `test_phase1.py`)**
  * Defines 8 tools (`database_query`, `document_search`, `calendar_lookup`, `email_draft`, `ticket_creation`, `status_lookup`, `threshold_check`, `audit_log_lookup`) with Pydantic input/output validation.
  * Runs tool-calling loops against local Ollama (`llama3.2`) with handling for invalid arguments, tool timeouts, and unknown/malformed tool calls.
* **Phase 2 — Model Context Protocol (`mcp_server.py`, `mcp_client.py`, `test_phase2.py`)**
  * Exposes mock CRM and ticketing tools (`crm_get_customer`, `ticket_get_status`, `ticket_create`) over `stdio` using the `mcp` Python SDK.
  * Enforces token authentication, role allow-lists (`ops_analyst`, `compliance_lead`), and per-tool permissions.
* **Phase 3 — LangGraph Orchestration & Evaluation (`orchestrator.py`, `eval_and_trace.py`, `test_phase3.py`)**
  * Implements a LangGraph `StateGraph` (`plan` -> `approval_gate` -> `act_and_observe`) with human-in-the-loop approval pause/resume, `max_steps` budget enforcement, failure-taxonomy classification, and optional Langfuse trace emission.

## Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt