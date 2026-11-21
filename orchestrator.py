import json
from typing import Any, Literal, TypedDict, cast
from langgraph.graph import END, START, StateGraph
from agent_runner import execute_tool_safely


STATE_CHANGING_TOOLS = {"ticket_creation", "email_draft"}


class WorkflowState(TypedDict):
    goal: str
    step_count: int
    max_steps: int
    pending_tool: dict[str, Any] | None
    approved: bool | None
    observations: list[dict[str, Any]]
    status: Literal["running", "awaiting_approval", "completed", "max_steps_reached"]
    summary: str


def plan_node(state: WorkflowState) -> dict[str, Any]:
    """Decides the next action based on current state and step budget."""
    step = state["step_count"] + 1
    if step > state["max_steps"]:
        return {
            "step_count": step,
            "status": "max_steps_reached",
            "summary": f"Stopped after reaching max_steps ({state['max_steps']}).",
        }

    # If a tool action is already approved, execute it next
    if state.get("pending_tool") and state.get("approved") is True:
        return {"step_count": step, "status": "running"}

    # Deterministic initial inspection plan for account/threshold workflows if no observations yet
    if not state.get("observations"):
        planned = {
            "tool": "database_query",
            "args": {"account_id": "ACC-102"},
        }
        return {
            "step_count": step,
            "pending_tool": planned,
            "approved": None,
            "status": "running",
        }

    # After account lookup, check threshold if not done yet
    executed_tools = {obs.get("tool") for obs in state.get("observations", [])}
    if "threshold_check" not in executed_tools:
        planned = {
            "tool": "threshold_check",
            "args": {"amount": 25000.0, "threshold": 10000.0, "label": "wire_review"},
        }
        return {
            "step_count": step,
            "pending_tool": planned,
            "approved": None,
            "status": "running",
        }

    # If threshold exceeded and no ticket created yet, propose a state-changing ticket_creation
    thresh_obs = next(
        (obs for obs in state.get("observations", []) if obs.get("tool") == "threshold_check"),
        None,
    )
    exceeded = bool(thresh_obs and thresh_obs.get("result", {}).get("exceeds_threshold"))
    if exceeded and "ticket_creation" not in executed_tools:
        planned = {
            "tool": "ticket_creation",
            "args": {
                "title": "Secondary compliance review for ACC-102",
                "priority": "high",
                "description": "Account balance/transfer exceeded EUR 10,000 threshold.",
                "account_id": "ACC-102",
            },
        }
        return {
            "step_count": step,
            "pending_tool": planned,
            "approved": False,
            "status": "awaiting_approval",
        }

    return {
        "step_count": step,
        "pending_tool": None,
        "status": "completed",
        "summary": "Workflow finished all planned checks.",
    }


def approval_gate_node(state: WorkflowState) -> dict[str, Any]:
    """Flags state-changing tools for explicit human approval before execution."""
    pending = state.get("pending_tool")
    if not pending:
        return {"status": "running"}

    tool_name = pending.get("tool", "")
    if tool_name in STATE_CHANGING_TOOLS and state.get("approved") is not True:
        return {
            "status": "awaiting_approval",
            "summary": f"Human approval required before executing '{tool_name}'.",
        }

    return {"status": "running"}


def act_and_observe_node(state: WorkflowState) -> dict[str, Any]:
    """Executes the pending tool and appends the observation to workflow state."""
    pending = state.get("pending_tool")
    if not pending:
        return {"status": "completed"}

    tool_name = pending["tool"]
    tool_args = pending.get("args", {})
    result = execute_tool_safely(tool_name, tool_args)

    new_obs = list(state.get("observations", []))
    new_obs.append({
        "step": state["step_count"],
        "tool": tool_name,
        "args": tool_args,
        "result": result,
    })

    return {
        "observations": new_obs,
        "pending_tool": None,
        "approved": None,
        "status": "running",
    }


def route_after_plan(state: WorkflowState) -> str:
    if state.get("status") in {"completed", "max_steps_reached"}:
        return "end"
    pending = state.get("pending_tool") or {}
    if pending.get("tool") in STATE_CHANGING_TOOLS and state.get("approved") is not True:
        return "await_approval"
    return "act"


def build_workflow_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("plan", plan_node)
    builder.add_node("approval_gate", approval_gate_node)
    builder.add_node("act_and_observe", act_and_observe_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "act": "act_and_observe",
            "await_approval": "approval_gate",
            "end": END,
        },
    )
    builder.add_edge("approval_gate", END)
    builder.add_edge("act_and_observe", "plan")

    return builder.compile()


def run_orchestrated_workflow(
    goal: str,
    max_steps: int = 6,
    approve_pending: bool = False,
    prior_state: WorkflowState | None = None,
) -> WorkflowState:
    graph = build_workflow_graph()
    if prior_state is not None:
        initial_state: WorkflowState = {
            **prior_state,
            "approved": approve_pending,
            "max_steps": max_steps,
        }
    else:
        initial_state = {
            "goal": goal,
            "step_count": 0,
            "max_steps": max_steps,
            "pending_tool": None,
            "approved": approve_pending if approve_pending else None,
            "observations": [],
            "status": "running",
            "summary": "",
        }

    return cast(WorkflowState, graph.invoke(initial_state))


if __name__ == "__main__":
    paused = run_orchestrated_workflow("Review ACC-102 against EUR 10,000 threshold.", max_steps=6)
    print("--- Paused for Human Approval ---")
    print(json.dumps(paused, indent=2))

    if paused["status"] == "awaiting_approval":
        resumed = run_orchestrated_workflow(
            paused["goal"],
            max_steps=6,
            approve_pending=True,
            prior_state=paused,
        )
        print("\n--- Resumed After Approval ---")
        print(json.dumps(resumed, indent=2))