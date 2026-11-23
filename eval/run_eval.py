"""Agent evaluation: runs every scenario through the real agent, MCP server and model,
then scores task completion, tool-selection accuracy and step efficiency, and counts
failures by type.

    python eval/run_eval.py                  # all scenarios, model from config/agent.yaml
    python eval/run_eval.py --model qwen2.5  # another Ollama model
    python eval/run_eval.py --only threshold-over,no-permission

Writes results/eval_report.json and results/eval_report.md.
"""

import argparse
import asyncio
import copy
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import Agent, load_config, load_prompt, summary  # noqa: E402
from llm import LLM, OllamaLLM  # noqa: E402
from mcp_host import MCPToolHost  # noqa: E402

TOKENS = {"ops": "tok-readonly-ops", "lead": "tok-compliance-lead"}


def load_scenarios(path: Path = ROOT / "eval" / "scenarios.yaml") -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text())["scenarios"]


def score(scn: dict[str, Any], state: dict[str, Any], seconds: float) -> dict[str, Any]:
    """Scores one finished run against its scenario."""
    traj = state.get("trajectory", [])
    attempted = [e["tool"] for e in traj if e["type"] == "tool_call"]
    executed_ok = {e["tool"] for e in traj if e["type"] == "tool_result" and e.get("ok")}
    expected = list(scn.get("expected_tools", []))
    forbidden = set(scn.get("forbidden_tools", []))

    # Tool selection: a call is correct if it is an expected tool, the first time it is used.
    seen: set[str] = set()
    correct = 0
    for tool in attempted:
        if tool in expected and tool not in seen:
            correct += 1
        seen.add(tool)
    precision = correct / len(attempted) if attempted else (1.0 if not expected else 0.0)
    recall = len(set(expected) & set(attempted)) / len(expected) if expected else 1.0

    optimal = scn.get("optimal_steps", len(expected))
    if not attempted:
        efficiency = 1.0 if optimal == 0 else 0.0
    else:
        efficiency = min(1.0, optimal / len(attempted)) if optimal else 0.0

    answer = (state.get("final_answer") or "").lower()
    groups = scn.get("answer_must_contain", [])
    answer_ok = all(any(alt.lower() in answer for alt in group) for group in groups)
    missing = [t for t in expected if t not in attempted]
    forbidden_hit = sorted(forbidden & executed_ok)
    status = state.get("status")
    status_ok = status == "completed"

    if status == "budget_exceeded":
        failure = "budget_exceeded"
    elif status == "failed":
        failure = f"stopped_by_guard:{state.get('failure')}"
    elif status == "awaiting_approval":
        failure = "stuck_awaiting_approval"
    elif forbidden_hit:
        failure = "forbidden_action"
    elif missing:
        failure = "missed_tool"
    elif not answer_ok:
        failure = "wrong_or_incomplete_answer"
    else:
        failure = "none"

    return {
        "id": scn["id"], "category": scn.get("category", ""),
        "completed": failure == "none",
        "failure": failure,
        "status": status,
        "tool_precision": round(precision, 3),
        "tool_recall": round(recall, 3),
        "step_efficiency": round(efficiency, 3),
        "tools_attempted": attempted,
        "expected_tools": expected,
        "missing_tools": missing,
        "forbidden_executed": forbidden_hit,
        "wrong_tool_calls": len(attempted) - correct,
        "guard_events": [e["error_type"] for e in traj if e["type"] == "guard"],
        "tool_errors": [e["error_type"] for e in traj if e["type"] == "tool_result" and not e.get("ok")],
        "llm_calls": state.get("llm_calls", 0),
        "tokens": state.get("prompt_tokens", 0) + state.get("completion_tokens", 0),
        "cost_usd": round(state.get("cost_usd", 0.0), 8),
        "seconds": round(seconds, 2),
        "final_answer": state.get("final_answer", ""),
    }


async def run_scenario(scn: dict[str, Any], host: MCPToolHost, llm: LLM, config: dict[str, Any],
                       prompt: dict[str, str]) -> dict[str, Any]:
    agent = Agent(host, llm, config, prompt)
    thread = f"eval-{scn['id']}-{time.time_ns()}"
    start = time.time()
    state = await agent.run(thread, goal=scn["goal"])
    pauses = 0
    while state.get("status") == "awaiting_approval" and pauses < 3:
        pauses += 1
        approve = scn.get("approval", "approve") == "approve"
        state = await agent.run(thread, decision={"approved": approve, "reason": "" if approve else "Not needed (eval)"})
    return score(scn, state, time.time() - start)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    mean = lambda key: round(sum(r[key] for r in results) / n, 3) if n else 0.0  # noqa: E731
    by_category: dict[str, dict[str, Any]] = {}
    for r in results:
        c = by_category.setdefault(r["category"], {"scenarios": 0, "completed": 0})
        c["scenarios"] += 1
        c["completed"] += int(r["completed"])
    secs = sorted(r["seconds"] for r in results)
    return {
        "scenarios": n,
        "task_completion_rate": round(sum(r["completed"] for r in results) / n, 3) if n else 0.0,
        "tool_selection_precision": mean("tool_precision"),
        "tool_selection_recall": mean("tool_recall"),
        "step_efficiency": mean("step_efficiency"),
        "wrong_tool_calls": sum(r["wrong_tool_calls"] for r in results),
        "failures": dict(Counter(r["failure"] for r in results if r["failure"] != "none")),
        "guard_events": dict(Counter(g for r in results for g in r["guard_events"])),
        "tool_errors": dict(Counter(t for r in results for t in r["tool_errors"])),
        "by_category": by_category,
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 6),
        "avg_tokens_per_task": round(sum(r["tokens"] for r in results) / n) if n else 0,
        "p50_seconds": secs[n // 2] if n else 0,
        "p95_seconds": secs[min(n - 1, int(n * 0.95))] if n else 0,
    }


async def evaluate(scenarios: list[dict[str, Any]], make_llm: Callable[[dict[str, Any]], LLM],
                   config: dict[str, Any], prompt: dict[str, str],
                   progress: Callable[[str], None] = print,
                   on_result: Callable[[list[dict[str, Any]]], None] | None = None) -> dict[str, Any]:
    """Runs every scenario. One scenario crashing (e.g. the model server failing) is retried once,
    then recorded as an `error` result, so it never stops the rest of the exam."""
    results: list[dict[str, Any]] = []
    order = {s["id"]: i for i, s in enumerate(scenarios)}
    for role in ("ops", "lead"):
        todo = [s for s in scenarios if s.get("role", "ops") == role]
        if not todo:
            continue
        async with MCPToolHost(TOKENS[role], timeout_seconds=config["tool_timeout_seconds"]) as host:
            for scn in todo:
                r = None
                for attempt in (1, 2):
                    try:
                        r = await run_scenario(scn, host, make_llm(scn), config, prompt)
                        break
                    except Exception as exc:  # noqa: BLE001 - the exam must survive one bad scenario
                        message = f"{type(exc).__name__}: {str(exc)[:200]}"
                        if attempt == 1:
                            progress(f"  retry {scn['id']}: {message}")
                            await asyncio.sleep(5)
                        else:
                            r = error_result(scn, message)
                progress(f"  {'PASS' if r['completed'] else 'FAIL'}  {r['id']:<24} {r['failure']:<28} "
                         f"tools={r['tools_attempted']}  {r['seconds']}s")
                if not r["completed"]:
                    progress(f"        answer: {(r['final_answer'] or '(none)')[:160]!r}")
                results.append(r)
                if on_result:
                    on_result(sorted(results, key=lambda x: order[x["id"]]))
    results.sort(key=lambda r: order[r["id"]])
    return {"summary": aggregate(results), "results": results}


def error_result(scn: dict[str, Any], message: str) -> dict[str, Any]:
    """A scenario that could not run at all (counts as not completed)."""
    return {
        "id": scn["id"], "category": scn.get("category", ""), "completed": False,
        "failure": "error", "status": "error", "error": message,
        "tool_precision": 0.0, "tool_recall": 0.0, "step_efficiency": 0.0,
        "tools_attempted": [], "expected_tools": list(scn.get("expected_tools", [])),
        "missing_tools": list(scn.get("expected_tools", [])), "forbidden_executed": [],
        "wrong_tool_calls": 0, "guard_events": [], "tool_errors": [],
        "llm_calls": 0, "tokens": 0, "cost_usd": 0.0, "seconds": 0.0, "final_answer": "",
    }


def to_markdown(report: dict[str, Any]) -> str:
    s, meta = report["summary"], report["meta"]
    lines = [
        f"# Agent evaluation: {meta['model']} (prompt {meta['prompt_version']})",
        "",
        f"Run {meta['finished_at']} on {s['scenarios']} of {meta.get('scenarios_planned', s['scenarios'])} scenarios"
        + ("." if meta.get("complete", True) else " (still running or stopped early)."),
        "",
        "| Metric | Result |",
        "| --- | --- |",
        f"| Task completion rate | {s['task_completion_rate']:.0%} |",
        f"| Tool-selection precision | {s['tool_selection_precision']:.0%} |",
        f"| Tool-selection recall | {s['tool_selection_recall']:.0%} |",
        f"| Step efficiency | {s['step_efficiency']:.0%} |",
        f"| Wrong tool calls | {s['wrong_tool_calls']} |",
        f"| Latency per task (P50 / P95) | {s['p50_seconds']}s / {s['p95_seconds']}s |",
        f"| Avg tokens per task | {s['avg_tokens_per_task']} |",
        f"| Total hosted-equivalent cost | ${s['total_cost_usd']} |",
        "",
        "## Failures by type",
        "",
        "| Failure | Count |",
        "| --- | --- |",
        *[f"| {k} | {v} |" for k, v in sorted(s["failures"].items(), key=lambda kv: -kv[1])],
        *(["| (none) | 0 |"] if not s["failures"] else []),
        "",
        "## Per scenario",
        "",
        "| Scenario | Result | Failure | Tools called | Precision | Efficiency |",
        "| --- | --- | --- | --- | --- | --- |",
        *[f"| {r['id']} | {'pass' if r['completed'] else 'FAIL'} | {r['failure']} | "
          f"{', '.join(r['tools_attempted']) or '-'} | {r['tool_precision']:.0%} | {r['step_efficiency']:.0%} |"
          for r in report["results"]],
        "",
        "## What the agent said when it failed",
        "",
        *[f"- **{r['id']}** ({r['failure']}): {(r.get('error') or r['final_answer'] or '(no answer)').strip()[:300]}"
          for r in report["results"] if not r["completed"]],
        *(["- (no failures)"] if all(r["completed"] for r in report["results"]) else []),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="Ollama model (default: config/agent.yaml)")
    ap.add_argument("--only", help="comma-separated scenario IDs")
    ap.add_argument("--out", default=str(ROOT / "results"), help="output folder")
    args = ap.parse_args()

    config = copy.deepcopy(load_config())
    if args.model:
        config["model"] = args.model
    prompt = load_prompt()
    scenarios = load_scenarios()
    if args.only:
        wanted = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in wanted]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": config["model"], "prompt_version": prompt["version"], "budgets": config["budgets"]}

    def save(results: list[dict[str, Any]], finished: bool) -> dict[str, Any]:
        report = {"summary": aggregate(results), "results": results, "meta": {
            **meta, "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "complete": finished, "scenarios_planned": len(scenarios)}}
        (out / "eval_report.json").write_text(json.dumps(report, indent=2))
        (out / "eval_report.md").write_text(to_markdown(report))
        return report

    print(f"Evaluating {config['model']} on {len(scenarios)} scenarios...")
    result = asyncio.run(evaluate(scenarios, lambda _: OllamaLLM(config["model"], config["temperature"]),
                                  config, prompt, on_result=lambda rs: save(rs, finished=False)))
    report = save(result["results"], finished=True)
    s = report["summary"]
    print(f"\nTask completion {s['task_completion_rate']:.0%} | tool precision {s['tool_selection_precision']:.0%} | "
          f"recall {s['tool_selection_recall']:.0%} | step efficiency {s['step_efficiency']:.0%}")
    print(f"Failures: {s['failures'] or 'none'}")
    print(f"Wrote {out / 'eval_report.md'}")


if __name__ == "__main__":
    main()
