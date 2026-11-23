"""Control panel back end: runs agent tasks and streams their trajectory to the browser.

    python server.py            # then open http://localhost:8000

POST /api/runs                 start a task {goal, role} -> {run_id}
GET  /api/runs/{id}/events     Server-Sent Events: every step as it happens
POST /api/runs/{id}/decision   approve or reject a paused action {approved, reason}
GET  /api/runs/{id}            current summary
GET  /api/meta                 roles, example goals, budgets, pricing

Each run gets its own MCP connection with the chosen role's token, so the panel
shows exactly what that role is allowed to do.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import Agent, load_config, load_prompt, summary
from llm import LLM, OllamaLLM
from mcp_host import MCPToolHost

ROOT = Path(__file__).resolve().parent
PANEL_DIST = ROOT / "panel" / "dist"
ROLES = {
    "ops": {"label": "Ops analyst (read-only)", "token": "tok-readonly-ops"},
    "lead": {"label": "Compliance lead", "token": "tok-compliance-lead"},
}
TERMINAL = {"completed", "budget_exceeded", "failed"}


@dataclass
class Run:
    id: str
    goal: str
    role: str
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    summary: dict[str, Any] = field(default_factory=dict)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    decision: asyncio.Future | None = None

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self.changed.set()


class StartRun(BaseModel):
    goal: str = Field(..., min_length=3, max_length=1000)
    role: str = "lead"


class Decision(BaseModel):
    approved: bool
    reason: str = Field(default="", max_length=300)


def create_app(make_llm: Callable[[dict[str, Any]], LLM] | None = None) -> FastAPI:
    app = FastAPI(title="Agent control panel")
    config, prompt = load_config(), load_prompt()
    make_llm = make_llm or (lambda cfg: OllamaLLM(cfg["model"], cfg["temperature"]))
    runs: dict[str, Run] = {}

    async def drive(run: Run) -> None:
        try:
            async with MCPToolHost(ROLES[run.role]["token"], timeout_seconds=config["tool_timeout_seconds"]) as host:
                run.emit({"type": "run_started", "t": time.time(), "role": run.role,
                          "tools": sorted(host.tool_names),
                          "gated_tools": sorted(n for n in host.tool_names if host.needs_approval(n)),
                          "budgets": config["budgets"], "pricing": config["pricing"],
                          "model": config["model"], "prompt_version": prompt["version"]})
                agent = Agent(host, make_llm(config), config, prompt)
                stream = agent.stream(run.id, goal=run.goal)
                while True:
                    async for ev in stream:
                        run.emit(ev)
                    state = await agent.snapshot(run.id)
                    run.status, run.summary = state["status"], summary(state)
                    run.emit({"type": "state", "t": time.time(), **run.summary})
                    if run.status != "awaiting_approval":
                        break
                    run.decision = asyncio.get_running_loop().create_future()
                    decision = await run.decision
                    run.decision = None
                    stream = agent.stream(run.id, decision=decision)
        except Exception as exc:  # surface errors (e.g. Ollama not running) in the panel
            run.status = "failed"
            run.emit({"type": "error", "t": time.time(), "text": f"{type(exc).__name__}: {exc}"})
        finally:
            run.emit({"type": "done", "t": time.time(), "status": run.status})

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        examples = yaml.safe_load((ROOT / "eval" / "scenarios.yaml").read_text())["scenarios"]
        pick = ["check-then-ticket", "check-no-ticket-needed", "email-rejected", "account-and-kyc",
                "no-permission", "slow-backend"]
        by_id = {s["id"]: s for s in examples}
        return {
            "roles": [{"id": k, "label": v["label"]} for k, v in ROLES.items()],
            "examples": [{"goal": by_id[i]["goal"], "role": by_id[i]["role"]} for i in pick if i in by_id],
            "model": config["model"], "prompt_version": prompt["version"],
        }

    @app.post("/api/runs")
    async def start(body: StartRun) -> dict[str, str]:
        if body.role not in ROLES:
            raise HTTPException(400, f"role must be one of {list(ROLES)}")
        run = Run(id=uuid.uuid4().hex[:12], goal=body.goal.strip(), role=body.role)
        runs[run.id] = run
        asyncio.create_task(drive(run))
        return {"run_id": run.id}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = runs.get(run_id) or _missing()
        return {"id": run.id, "goal": run.goal, "role": run.role, "status": run.status, "summary": run.summary}

    @app.post("/api/runs/{run_id}/decision")
    def decide(run_id: str, body: Decision) -> dict[str, str]:
        run = runs.get(run_id) or _missing()
        if run.decision is None or run.decision.done():
            raise HTTPException(409, "This run is not waiting for approval.")
        run.decision.set_result({"approved": body.approved, "reason": body.reason})
        return {"ok": "resumed"}

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str) -> StreamingResponse:
        run = runs.get(run_id) or _missing()

        async def gen():
            sent = 0
            while True:
                while sent < len(run.events):
                    ev = run.events[sent]
                    sent += 1
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
                    if ev["type"] == "done":
                        return
                run.changed.clear()
                try:
                    await asyncio.wait_for(run.changed.wait(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if PANEL_DIST.exists():
        app.mount("/assets", StaticFiles(directory=PANEL_DIST / "assets"), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(PANEL_DIST / "index.html")

    return app


def _missing() -> Run:
    raise HTTPException(404, "Unknown run")


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
