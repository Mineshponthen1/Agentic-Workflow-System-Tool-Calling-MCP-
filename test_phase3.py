from orchestrator import run_orchestrated_workflow


def test_workflow_pauses_for_approval_on_ticket_creation():
    paused = run_orchestrated_workflow(
        "Review ACC-102 against EUR 10,000 threshold.",
        max_steps=6,
    )
    assert paused["status"] == "awaiting_approval"
    assert paused["pending_tool"] is not None
    assert paused["pending_tool"]["tool"] == "ticket_creation"
    assert len(paused["observations"]) == 2


def test_workflow_resumes_and_completes_when_approved():
    paused = run_orchestrated_workflow(
        "Review ACC-102 against EUR 10,000 threshold.",
        max_steps=6,
    )
    resumed = run_orchestrated_workflow(
        paused["goal"],
        max_steps=6,
        approve_pending=True,
        prior_state=paused,
    )
    assert resumed["status"] == "completed"
    assert resumed["pending_tool"] is None
    assert any(obs["tool"] == "ticket_creation" for obs in resumed["observations"])


def test_workflow_respects_max_steps_budget():
    stopped = run_orchestrated_workflow(
        "Review ACC-102 against EUR 10,000 threshold.",
        max_steps=1,
    )
    assert stopped["status"] == "max_steps_reached"
    assert stopped["step_count"] > 1