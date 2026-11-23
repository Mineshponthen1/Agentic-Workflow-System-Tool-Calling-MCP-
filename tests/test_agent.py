"""The agent loop with a scripted model: planning, passing results forward, every guardrail."""

import asyncio
import copy

from agent import Agent, load_config, load_prompt, summary
from llm import ScriptedLLM, answer, call, last_tool_result
from mcp_host import MCPToolHost


def run_agent(script, goal="Test goal", token="tok-compliance-lead", decisions=(), config_patch=None, timeout=3.0):
    """Runs a scripted task to the end, answering approval pauses with `decisions` in order."""
    async def go():
        cfg = copy.deepcopy(load_config())
        for key, value in (config_patch or {}).items():
            cfg["budgets"][key] = value
        async with MCPToolHost(token, timeout_seconds=timeout) as host:
            agent = Agent(host, ScriptedLLM(script), cfg, load_prompt())
            state = await agent.run("t", goal=goal)
            for decision in decisions:
                assert state["status"] == "awaiting_approval"
                state = await agent.run("t", decision=decision)
            return state
    return asyncio.run(go())


def events(state, kind):
    return [e for e in state["trajectory"] if e["type"] == kind]


def test_results_flow_forward_and_write_waits_for_approval():
    script = [
        call("database_query", account_id="ACC-102"),
        lambda m: call("threshold_check", amount=last_tool_result(m)["balance_eur"], threshold=10000),
        call("ticket_creation", title="Review ACC-102", priority="high", description="Over the limit", account_id="ACC-102"),
        lambda m: answer(f"Opened {last_tool_result(m)['ticket_id']}."),
    ]
    paused = run_agent(script)
    assert paused["status"] == "awaiting_approval"
    assert summary(paused)["pending_approval"]["name"] == "ticket_creation"
    threshold_call = [e for e in events(paused, "tool_call") if e["tool"] == "threshold_check"][0]
    assert threshold_call["args"]["amount"] == 25000.0  # taken from the lookup, not typed in

    done = run_agent(copy.deepcopy(script), decisions=[{"approved": True}])
    assert done["status"] == "completed"
    assert done["final_answer"].startswith("Opened TKT-")
    assert done["steps"] == 3


def test_rejected_action_is_not_executed():
    script = [
        call("ticket_creation", title="Close account", priority="low", description="Customer asked"),
        answer("The ticket was not created because the reviewer rejected it."),
    ]
    state = run_agent(script, decisions=[{"approved": False, "reason": "not needed"}])
    assert state["status"] == "completed"
    assert not [e for e in events(state, "tool_result") if e["tool"] == "ticket_creation"]
    assert events(state, "approval")[0]["approved"] is False


def test_unknown_tool_is_caught_and_model_recovers():
    script = [
        call("transfer_money", amount=500),
        call("database_query", account_id="ACC-101"),
        answer("Clara Weber, active."),
    ]
    state = run_agent(script)
    assert state["status"] == "completed"
    assert events(state, "guard")[0]["error_type"] == "unknown_tool"
    assert state["steps"] == 1  # the bad call never reached a tool


def test_hidden_tool_counts_as_unknown_for_read_only_role():
    script = [call("ticket_creation", title="x", priority="high", description="hello"), answer("I can't create tickets.")]
    state = run_agent(script, token="tok-readonly-ops")
    assert events(state, "guard")[0]["error_type"] == "unknown_tool"
    assert not events(state, "approval_request")


def test_repeated_identical_call_is_blocked():
    script = [
        call("database_query", account_id="ACC-101"),
        call("database_query", account_id="ACC-101"),
        answer("Done."),
    ]
    state = run_agent(script)
    assert events(state, "guard")[0]["error_type"] == "repeated_call"
    assert state["steps"] == 1


def test_too_many_guard_failures_stop_the_task():
    script = [call("nope_1"), call("nope_2"), call("nope_3"), answer("never reached")]
    state = run_agent(script)
    assert state["status"] == "failed" and state["failure"] == "unknown_tool"


def test_step_budget_stops_a_runaway_agent():
    script = [call("database_query", account_id=f"ACC-10{i}") for i in range(1, 4)] + [answer("never reached")]
    state = run_agent(script, config_patch={"max_steps": 2})
    assert state["status"] == "budget_exceeded" and state["steps"] == 2


def test_cost_budget_stops_the_task():
    script = [call("database_query", account_id="ACC-101"), answer("never reached")]
    state = run_agent(script, config_patch={"max_cost_usd": 0.000001})
    assert state["status"] == "budget_exceeded"
    assert "max_cost_usd" in state["final_answer"]


def test_tool_timeout_is_reported_to_the_model():
    script = [
        call("database_query", account_id="ACC-999"),
        lambda m: answer(f"Lookup failed: {last_tool_result(m)['error']}"),
    ]
    state = run_agent(script, timeout=1.0)
    assert events(state, "tool_result")[0]["error_type"] == "timeout"
    assert "timed out" in state["final_answer"]


def test_validation_error_goes_back_to_the_model():
    script = [
        call("calendar_lookup", calendar_name="compliance", date_iso="next friday"),
        call("calendar_lookup", calendar_name="compliance", date_iso="2026-12-04"),
        answer("Free at 09:30."),
    ]
    state = run_agent(script)
    results = events(state, "tool_result")
    assert results[0]["error_type"] == "validation_error" and results[1]["ok"]
    assert state["status"] == "completed"
