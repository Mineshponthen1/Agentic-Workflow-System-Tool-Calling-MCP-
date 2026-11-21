from tools import TOOLS, database_query, ticket_creation
from agent_runner import execute_tool_safely, run_agent_step


def test_eight_tools_registered():
    assert len(TOOLS) == 8


def test_database_query_valid_and_timeout():
    ok = database_query({"account_id": "ACC-102"})
    assert ok["found"] is True
    assert ok["balance_eur"] == 25000.0

    timed_out = database_query({"account_id": "ACC-102", "simulate_timeout": True})
    assert timed_out["status"] == "timeout"


def test_invalid_arguments_caught():
    bad = ticket_creation({"title": "Test", "priority": "super-urgent", "description": "Bad priority"})
    assert "error" in bad


def test_wrong_or_unknown_tool_handled_safely():
    res = execute_tool_safely("non_existent_tool", {"x": 1})
    assert "error" in res


def test_agent_loop_with_fake_ollama():
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "database_query",
                                "arguments": {"account_id": "ACC-101"},
                            }
                        }
                    ],
                }
            }
        return {"message": {"content": "Account ACC-101 belongs to Clara Weber and is active."}}

    result = run_agent_step("Check ACC-101", chat_fn=fake_chat)
    assert result["status"] == "completed"
    assert "Clara Weber" in result["final_answer"]
    assert len(result["tool_events"]) == 1