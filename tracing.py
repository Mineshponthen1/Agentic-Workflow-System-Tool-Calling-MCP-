"""Optional Langfuse tracing for agent runs (Langfuse Python SDK v3+, OpenTelemetry-based).

Tracing is on only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set
(LANGFUSE_HOST defaults to Langfuse Cloud; point it at a self-hosted instance).

A task that pauses for approval runs in two or more segments. Each segment is
written under the same trace ID (derived from the thread ID), so the whole task
shows as one trace: one generation per model decision, one tool observation per
tool call, a guardrail observation per blocked call, and the approval decisions.

Monitoring never breaks the product: any tracing error is printed and ignored.
"""

import os
import sys
from typing import Any

_client: Any = None


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _get() -> Any:
    global _client
    if _client is None:
        from langfuse import get_client

        _client = get_client()
    return _client


def record_segment(thread_id: str, goal: str, events: list[dict[str, Any]],
                   summary: dict[str, Any], model: str) -> str | None:
    """Writes one run segment to Langfuse. Returns the trace ID, or None if tracing is off."""
    if not enabled() or not events:
        return None
    try:
        from langfuse import Langfuse

        lf = _get()
        trace_id = Langfuse.create_trace_id(seed=thread_id)
        with lf.start_as_current_observation(
            trace_context={"trace_id": trace_id}, name="agent-task", as_type="agent",
            input=goal, output=summary.get("final_answer") or summary.get("status"),
            metadata={"thread_id": thread_id, "status": summary.get("status"),
                      "prompt_version": summary.get("prompt_version"), "failure": summary.get("failure")},
        ) as root:
            for ev in events:
                kind = ev["type"]
                if kind in ("tool_call", "final"):
                    gen = root.start_observation(
                        name="llm-decision", as_type="generation", model=model,
                        output={"tool": ev.get("tool"), "args": ev.get("args")} if kind == "tool_call" else ev.get("text"),
                        usage_details={"input": ev.get("prompt_tokens", 0), "output": ev.get("completion_tokens", 0)},
                    )
                    gen.end()
                elif kind == "tool_result":
                    obs = root.start_observation(
                        name=ev["tool"], as_type="tool", output=ev.get("result"),
                        level="DEFAULT" if ev.get("ok") else "ERROR",
                        status_message=ev.get("error_type") or None,
                    )
                    obs.end()
                elif kind == "guard":
                    obs = root.start_observation(name=f"guard:{ev.get('error_type')}", as_type="guardrail",
                                                 output=ev.get("text"), level="WARNING")
                    obs.end()
                elif kind in ("approval_request", "approval", "stopped"):
                    obs = root.start_observation(name=kind, as_type="span", output={k: v for k, v in ev.items() if k != "t"})
                    obs.end()
            root.score_trace(name="outcome", value=str(summary.get("status")), data_type="CATEGORICAL")
        lf.flush()
        return trace_id
    except Exception as exc:  # never let monitoring break the agent
        print(f"[tracing] skipped: {exc}", file=sys.stderr)
        return None
