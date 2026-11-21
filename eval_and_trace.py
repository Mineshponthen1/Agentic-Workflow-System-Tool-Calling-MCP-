import json
import os
import time
from typing import Any, Literal
from agent_runner import execute_tool_safely
from orchestrator import run_orchestrated_workflow


FailureCategory = Literal[
    "none",
    "validation_error",
    "timeout_error",
    "permission_denied",
    "budget_exceeded",
    "unexpected_outcome",
]


def classify_failure(payload: dict[str, Any]) -> FailureCategory:
    """Maps tool or workflow outputs into a standardized failure taxonomy."""
    err_text = json.dumps(payload).lower()
    if payload.get("status") == "max_steps_reached":
        return "budget_exceeded"
    if payload.get("status") == "timeout" or "timeout" in err_text or "timed out" in err_text:
        return "timeout_error"
    if payload.get("authorized") is False or "unauthorized" in err_text or "forbidden" in err_text:
        return "permission_denied"
    if "invalid input" in err_text or "validationerror" in err_text or "malformed" in err_text:
        return "validation_error"
    if payload.get("status") == "error" or ("error" in err_text and payload.get("error")):
        return "unexpected_outcome"
    return "none"


def emit_trace_event(name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Records a local trace event and optionally forwards to Langfuse if configured."""
    event = {
        "trace_name": name,
        "timestamp": round(time.time(), 3),
        "langfuse_enabled": bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")),
        "metadata": metadata,
    }
    if event["langfuse_enabled"]:
        try:
            from langfuse import Langfuse  # type: ignore

            lf = Langfuse()
            trace = lf.trace(name=name, metadata=metadata)
            event["langfuse_trace_id"] = getattr(trace, "id", None)
        except Exception as exc:
            event["langfuse_note"] = f"Langfuse skipped: {exc}"
    return event


def run_scenario_evaluations() -> dict[str, Any]:
    """Runs scripted scenarios covering normal execution, approval gates, budgets, and failure paths."""
    scenarios: list[dict[str, Any]] = []

    # Scenario 1: Happy-path account + threshold check + approval pause
    s1_state = run_orchestrated_workflow("Review ACC-102 against EUR 10,000 threshold.", max_steps=6)
    s1_cat = classify_failure(dict(s1_state))
    s1_passed = s1_state["status"] == "awaiting_approval" and s1_cat == "none"
    scenarios.append({
        "scenario": "account_threshold_approval_gate",
        "passed": s1_passed,
        "failure_category": s1_cat,
        "status": s1_state["status"],
    })

    # Scenario 2: Step-budget enforcement
    s2_state = run_orchestrated_workflow("Review ACC-102 against EUR 10,000 threshold.", max_steps=1)
    s2_cat = classify_failure(dict(s2_state))
    s2_passed = s2_state["status"] == "max_steps_reached" and s2_cat == "budget_exceeded"
    scenarios.append({
        "scenario": "max_steps_budget_enforcement",
        "passed": s2_passed,
        "failure_category": s2_cat,
        "status": s2_state["status"],
    })

    # Scenario 3: Tool timeout handling
    s3_res = execute_tool_safely("database_query", {"account_id": "ACC-102", "simulate_timeout": True})
    s3_cat = classify_failure(s3_res)
    s3_passed = s3_res.get("status") == "timeout" and s3_cat == "timeout_error"
    scenarios.append({
        "scenario": "tool_timeout_handling",
        "passed": s3_passed,
        "failure_category": s3_cat,
        "status": s3_res.get("status", "ok"),
    })

    # Scenario 4: Invalid tool argument validation
    s4_res = execute_tool_safely("ticket_creation", {"title": "Bad", "priority": "urgent", "description": "Invalid"})
    s4_cat = classify_failure(s4_res)
    s4_passed = "error" in s4_res and s4_cat == "validation_error"
    scenarios.append({
        "scenario": "invalid_argument_validation",
        "passed": s4_passed,
        "failure_category": s4_cat,
        "status": "rejected",
    })

    report = {
        "total_scenarios": len(scenarios),
        "passed_count": sum(1 for item in scenarios if item["passed"]),
        "results": scenarios,
    }
    report["trace"] = emit_trace_event(
        "phase3_scenario_eval",
        {"passed_count": report["passed_count"], "total": report["total_scenarios"]},
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run_scenario_evaluations(), indent=2))