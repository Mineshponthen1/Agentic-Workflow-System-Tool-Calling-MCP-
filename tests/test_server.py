"""The control panel API, with a scripted model."""

import time

from fastapi.testclient import TestClient

from llm import ScriptedLLM, answer, call
from server import create_app


def make_client(script):
    return TestClient(create_app(lambda cfg: ScriptedLLM(list(script))))


def wait_for(client, run_id, status, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] == status:
            return run
        time.sleep(0.2)
    raise AssertionError(f"run never reached {status}: {run}")


def test_meta_lists_roles_and_examples():
    with make_client([]) as client:
        meta = client.get("/api/meta").json()
    assert {r["id"] for r in meta["roles"]} == {"ops", "lead"}
    assert len(meta["examples"]) >= 4


def test_run_pauses_then_completes_after_approval():
    script = [call("ticket_creation", title="Review ACC-102", priority="high", description="Over the limit"),
              answer("Ticket opened.")]
    with make_client(script) as client:
        run_id = client.post("/api/runs", json={"goal": "Open a ticket for ACC-102", "role": "lead"}).json()["run_id"]
        wait_for(client, run_id, "awaiting_approval")
        assert client.post(f"/api/runs/{run_id}/decision", json={"approved": True}).status_code == 200
        done = wait_for(client, run_id, "completed")
        assert done["summary"]["final_answer"] == "Ticket opened."
        body = client.get(f"/api/runs/{run_id}/events").text
    for kind in ("run_started", "approval_request", "approval", "tool_result", "final", "done"):
        assert f'"type": "{kind}"' in body


def test_decision_refused_when_not_waiting():
    with make_client([answer("Hello.")]) as client:
        run_id = client.post("/api/runs", json={"goal": "Say hello", "role": "ops"}).json()["run_id"]
        wait_for(client, run_id, "completed")
        assert client.post(f"/api/runs/{run_id}/decision", json={"approved": True}).status_code == 409
        assert client.post("/api/runs", json={"goal": "Hi there", "role": "admin"}).status_code == 400
