"""The agent: a LangGraph plan -> guard -> (approve) -> act -> observe loop.

    plan ──► guard ──► act ──► plan ...         (read-only tools)
               │
               └──► approve ──► act ──► plan    (state-changing tools: pauses with interrupt())

- plan:    the model reads the goal and everything observed so far, then picks the
           next tool and its arguments, or gives the final answer.
- guard:   catches the model choosing a tool that does not exist (or that this
           caller may not see), unparseable arguments, and repeated identical calls.
           The error goes back to the model so it can correct itself; too many
           guard failures stop the task.
- approve: state-changing tools (the MCP server marks them not read-only) pause
           the graph until a human approves or rejects.
- act:     calls the tool through the MCP host, with a real timeout.

Hard budgets on tool steps, model calls and cost stop runaway tasks.
"""

import json
import operator
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Literal, TypedDict

import yaml
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import tracing
from llm import LLM
from mcp_host import MCPToolHost

ROOT = Path(__file__).resolve().parent
MAX_RESULT_CHARS = 1500

Status = Literal["running", "awaiting_approval", "completed", "budget_exceeded", "failed"]


def load_config(path: Path = ROOT / "config" / "agent.yaml") -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def load_prompt(path: Path = ROOT / "prompts" / "agent_system.yaml") -> dict[str, str]:
    return yaml.safe_load(path.read_text())


class AgentState(TypedDict, total=False):
    goal: str
    messages: list[dict[str, Any]]
    trajectory: Annotated[list[dict[str, Any]], operator.add]
    steps: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    guard_failures: int
    executed: list[str]
    pending: dict[str, Any] | None
    route: str
    status: Status
    final_answer: str
    failure: str | None
    prompt_version: str


def _event(kind: str, **data: Any) -> dict[str, Any]:
    return {"type": kind, "t": round(time.time(), 3), **data}


def _signature(name: str, args: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


def _short(text: str) -> str:
    text = text.split("\n    For further information")[0]
    text = text.replace("Error executing tool ", "")
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + " ...[truncated]"


@dataclass
class Agent:
    host: MCPToolHost
    llm: LLM
    config: dict[str, Any]
    prompt: dict[str, str]

    def __post_init__(self) -> None:
        self.budgets = self.config["budgets"]
        self.pricing = self.config["pricing"]
        self.graph = self._build().compile(checkpointer=InMemorySaver())

    # ---------- nodes ----------

    async def _plan(self, state: AgentState) -> dict[str, Any]:
        b = self.budgets
        for over, what in (
            (state["steps"] >= b["max_steps"], f"max_steps ({b['max_steps']})"),
            (state["llm_calls"] >= b["max_llm_calls"], f"max_llm_calls ({b['max_llm_calls']})"),
            (state["cost_usd"] >= b["max_cost_usd"], f"max_cost_usd (${b['max_cost_usd']})"),
        ):
            if over:
                return {
                    "status": "budget_exceeded", "failure": "budget_exceeded",
                    "final_answer": f"Stopped: hit the {what} budget before finishing.",
                    "trajectory": [_event("stopped", reason=f"budget: {what}")],
                }

        reply = await self.llm.chat(state["messages"], self.host.tool_definitions)
        cost = (reply.prompt_tokens * self.pricing["usd_per_million_input_tokens"]
                + reply.completion_tokens * self.pricing["usd_per_million_output_tokens"]) / 1_000_000
        usage = {
            "llm_calls": state["llm_calls"] + 1,
            "prompt_tokens": state["prompt_tokens"] + reply.prompt_tokens,
            "completion_tokens": state["completion_tokens"] + reply.completion_tokens,
            "cost_usd": state["cost_usd"] + cost,
        }

        if not reply.tool_calls:
            return {
                **usage, "status": "completed", "final_answer": reply.content.strip(), "pending": None,
                "messages": state["messages"] + [{"role": "assistant", "content": reply.content}],
                "trajectory": [_event("final", text=reply.content.strip(),
                                      prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens)],
            }

        tc = reply.tool_calls[0]  # one tool per step keeps approvals and step counts exact
        events = []
        if reply.content.strip():
            events.append(_event("thought", text=reply.content.strip()))
        events.append(_event("tool_call", tool=tc.name, args=tc.arguments, step=state["steps"] + 1,
                             prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens))
        return {
            **usage,
            "pending": {"name": tc.name, "args": tc.arguments},
            "messages": state["messages"] + [{
                "role": "assistant", "content": reply.content,
                "tool_calls": [{"function": {"name": tc.name, "arguments": tc.arguments}}],
            }],
            "trajectory": events,
        }

    def _guard(self, state: AgentState) -> dict[str, Any]:
        name, args = state["pending"]["name"], state["pending"]["args"]
        problem, kind = None, None
        if name not in self.host.tool_names:
            available = ", ".join(sorted(self.host.tool_names))
            problem, kind = f"Tool '{name}' is not available to you. Available tools: {available}.", "unknown_tool"
        elif "__unparseable__" in args:
            problem, kind = f"Your arguments for '{name}' were not valid JSON.", "validation_error"
        elif _signature(name, args) in state["executed"]:
            problem, kind = (f"You already called '{name}' with these exact arguments. "
                             "Use that earlier result instead of calling it again."), "repeated_call"

        if problem:
            failures = state["guard_failures"] + 1
            update: dict[str, Any] = {
                "guard_failures": failures, "pending": None,
                "messages": state["messages"] + [{"role": "tool", "tool_name": name, "content": json.dumps({"error": problem})}],
                "trajectory": [_event("guard", tool=name, error_type=kind, text=problem)],
            }
            if failures >= self.budgets["max_guard_failures"]:
                update.update(status="failed", failure=kind, route="end",
                              final_answer=f"Stopped: {failures} invalid tool calls in a row (last: {kind}).")
                update["trajectory"] = update["trajectory"] + [_event("stopped", reason=f"guard limit: {kind}")]
            else:
                update["route"] = "plan"
            return update

        if self.host.needs_approval(name):
            return {"route": "approve", "status": "awaiting_approval",
                    "trajectory": [_event("approval_request", tool=name, args=args)]}
        return {"route": "act"}

    def _approve(self, state: AgentState) -> dict[str, Any]:
        name, args = state["pending"]["name"], state["pending"]["args"]
        decision = interrupt({"tool": name, "args": args})
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        reason = (decision.get("reason") if isinstance(decision, dict) else None) or ""
        event = _event("approval", tool=name, approved=approved, reason=reason)
        if approved:
            return {"route": "act", "status": "running", "trajectory": [event]}
        note = f"A human reviewer rejected '{name}'" + (f": {reason}" if reason else "") + \
               ". Do not retry it. Tell the user it was not done."
        return {
            "route": "plan", "status": "running", "pending": None,
            "executed": state["executed"] + [_signature(name, args)],
            "messages": state["messages"] + [{"role": "tool", "tool_name": name, "content": json.dumps({"error": note})}],
            "trajectory": [event],
        }

    async def _act(self, state: AgentState) -> dict[str, Any]:
        name, args = state["pending"]["name"], state["pending"]["args"]
        outcome = await self.host.call(name, args)
        payload = outcome.result if outcome.ok else {"error": _short(str(outcome.result.get("error", "")))}
        update: dict[str, Any] = {
            "steps": state["steps"] + 1, "pending": None, "status": "running",
            "executed": state["executed"] + [_signature(name, args)],
            "messages": state["messages"] + [{"role": "tool", "tool_name": name, "content": json.dumps(payload, default=str)}],
            "trajectory": [_event("tool_result", tool=name, ok=outcome.ok, error_type=outcome.error_type, result=payload)],
        }
        if outcome.error_type == "validation_error":
            update["guard_failures"] = state["guard_failures"] + 1
        elif outcome.ok:
            update["guard_failures"] = 0
        return update

    # ---------- graph ----------

    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("guard", self._guard)
        g.add_node("approve", self._approve)
        g.add_node("act", self._act)
        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", lambda s: "end" if s["status"] in ("completed", "budget_exceeded", "failed") else "guard",
                                {"guard": "guard", "end": END})
        g.add_conditional_edges("guard", lambda s: s["route"], {"plan": "plan", "act": "act", "approve": "approve", "end": END})
        g.add_conditional_edges("approve", lambda s: s["route"], {"plan": "plan", "act": "act"})
        g.add_edge("act", "plan")
        return g

    # ---------- running ----------

    def initial_state(self, goal: str) -> AgentState:
        system = self.prompt["template"].format(today=date.today().isoformat())
        return {
            "goal": goal, "status": "running", "final_answer": "", "failure": None, "pending": None,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": goal}],
            "trajectory": [_event("goal", text=goal)],
            "steps": 0, "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
            "guard_failures": 0, "executed": [], "route": "", "prompt_version": self.prompt["version"],
        }

    async def stream(self, thread_id: str, goal: str | None = None,
                     decision: dict[str, Any] | None = None) -> AsyncIterator[dict[str, Any]]:
        """Runs (or resumes, with a decision) a task and yields trajectory events as they happen."""
        cfg = {"configurable": {"thread_id": thread_id}}
        inp: Any = self.initial_state(goal) if goal is not None else Command(resume=decision)
        segment: list[dict[str, Any]] = []
        if goal is not None:
            segment.append(inp["trajectory"][0])
            yield inp["trajectory"][0]
        async for update in self.graph.astream(inp, cfg, stream_mode="updates"):
            for node_update in update.values():
                if isinstance(node_update, dict):
                    for ev in node_update.get("trajectory", []):
                        segment.append(ev)
                        yield ev
        if tracing.enabled():
            state = await self.snapshot(thread_id)
            tracing.record_segment(thread_id, state.get("goal", ""), segment, summary(state), self.config["model"])

    async def run(self, thread_id: str, goal: str | None = None,
                  decision: dict[str, Any] | None = None) -> AgentState:
        async for _ in self.stream(thread_id, goal, decision):
            pass
        return await self.snapshot(thread_id)

    async def snapshot(self, thread_id: str) -> AgentState:
        state = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        return dict(state.values)  # type: ignore[return-value]


def summary(state: AgentState) -> dict[str, Any]:
    """The numbers a person (or the eval) cares about, without the message history."""
    return {
        "status": state.get("status"),
        "final_answer": state.get("final_answer"),
        "failure": state.get("failure"),
        "steps": state.get("steps"),
        "llm_calls": state.get("llm_calls"),
        "tokens": state.get("prompt_tokens", 0) + state.get("completion_tokens", 0),
        "cost_usd": round(state.get("cost_usd", 0.0), 8),
        "tools_called": [e["tool"] for e in state.get("trajectory", []) if e["type"] == "tool_call"],
        "pending_approval": state.get("pending") if state.get("status") == "awaiting_approval" else None,
        "prompt_version": state.get("prompt_version"),
    }
