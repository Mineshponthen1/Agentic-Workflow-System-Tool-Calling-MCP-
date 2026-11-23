"""Checks the evaluation harness itself: a perfect scripted agent must score 100% on every
scenario, and known mistakes must be scored as the right failure type."""

import asyncio
import copy

from agent import load_config, load_prompt
from eval.run_eval import evaluate, load_scenarios
from llm import ScriptedLLM, answer, call, last_tool_result

R = last_tool_result

# A reference trajectory per scenario: the shortest correct way to do each task.
REFERENCE = {
    "balance-lookup": [call("database_query", account_id="ACC-101"),
                       lambda m: answer(f"ACC-101 is {R(m)['status']} with EUR {R(m)['balance_eur']:,.0f}.")],
    "kyc-status": [call("crm_get_customer", account_id="ACC-103"), lambda m: answer(f"KYC is {R(m)['kyc_status']}.")],
    "policy-search": [call("document_search", query="transfers above 10,000", category="compliance"),
                      lambda m: answer(R(m)["matches"][0])],
    "ticket-status": [call("status_lookup", ticket_id="TKT-102"),
                      lambda m: answer(f"{R(m)['state']}, owned by {R(m)['owner']}.")],
    "audit-events": [call("audit_log_lookup", entity_id="ACC-103"), lambda m: answer("; ".join(R(m)["events"]))],
    "calendar-slots": [call("calendar_lookup", calendar_name="compliance", date_iso="2026-12-04"),
                       lambda m: answer(", ".join(R(m)["available_slots"]))],
    "threshold-over": [call("database_query", account_id="ACC-102"),
                       lambda m: call("threshold_check", amount=R(m)["balance_eur"], threshold=10000),
                       lambda m: answer(f"Yes, EUR {R(m)['amount']:,.0f} exceeds EUR 10,000.")],
    "threshold-under": [call("database_query", account_id="ACC-101"),
                        lambda m: call("threshold_check", amount=R(m)["balance_eur"], threshold=10000),
                        lambda m: answer(f"No, EUR {R(m)['amount']:,.0f} is below EUR 10,000.")],
    "arithmetic-in-code": [call("threshold_check", amount=9999.5, threshold=10000),
                           answer("No, EUR 9,999.50 is not above the limit.")],
    "account-and-kyc": [call("database_query", account_id="ACC-103"), call("crm_get_customer", account_id="ACC-103"),
                        answer("Account is restricted and KYC is expired.")],
    "kyc-and-policy": [call("crm_get_customer", account_id="ACC-103"),
                       call("document_search", query="KYC expired", category="compliance"),
                       answer("KYC is expired; policy says no transactions until KYC is refreshed.")],
    "audit-and-ticket": [call("audit_log_lookup", entity_id="ACC-102"), call("status_lookup", ticket_id="TKT-101"),
                         answer("A review was opened on 2026-11-05; TKT-101 is pending approval.")],
    "ticket-approved": [call("ticket_creation", title="Compliance review ACC-102", priority="high",
                             description="Over the EUR 10,000 limit", account_id="ACC-102"),
                        lambda m: answer(f"Opened {R(m)['ticket_id']}.")],
    "check-then-ticket": [call("database_query", account_id="ACC-102"),
                          lambda m: call("threshold_check", amount=R(m)["balance_eur"], threshold=10000),
                          call("ticket_creation", title="Secondary review ACC-102", priority="high",
                               description="Balance exceeds EUR 10,000", account_id="ACC-102"),
                          lambda m: answer(f"Over the limit, so I opened {R(m)['ticket_id']}.")],
    "check-no-ticket-needed": [call("database_query", account_id="ACC-101"),
                               lambda m: call("threshold_check", amount=R(m)["balance_eur"], threshold=10000),
                               answer("EUR 8,400 is below the limit, so no ticket is needed.")],
    "email-rejected": [call("email_draft", to="compliance@bank.example", subject="ACC-103 KYC expired",
                            body="The customer on ACC-103 has expired KYC."),
                       answer("The draft was not created: the reviewer rejected it.")],
    "no-permission": [answer("I don't have permission to create tickets with this role.")],
    "slow-backend": [call("database_query", account_id="ACC-999"),
                     answer("The account lookup timed out, so I could not get the details.")],
    "unknown-account": [call("database_query", account_id="ACC-404"), answer("Account ACC-404 was not found.")],
    "out-of-scope": [answer("I can't help with weather; I only have banking tools.")],
}


def run(scenarios, scripts, timeout=None):
    config = copy.deepcopy(load_config())
    if timeout:
        config["tool_timeout_seconds"] = timeout
    make = lambda scn: ScriptedLLM(copy.deepcopy(scripts[scn["id"]]))  # noqa: E731
    return asyncio.run(evaluate(scenarios, make, config, load_prompt(), progress=lambda _: None))


def test_every_scenario_has_a_reference_and_a_perfect_agent_scores_100():
    scenarios = load_scenarios()
    assert {s["id"] for s in scenarios} == set(REFERENCE)
    report = run(scenarios, REFERENCE, timeout=1.0)
    failed = [(r["id"], r["failure"]) for r in report["results"] if not r["completed"]]
    assert failed == []
    s = report["summary"]
    assert s["task_completion_rate"] == 1.0
    assert s["tool_selection_precision"] == 1.0 and s["tool_selection_recall"] == 1.0
    assert s["step_efficiency"] == 1.0


def test_mistakes_are_scored_as_the_right_failure():
    by_id = {s["id"]: s for s in load_scenarios()}
    bad = {
        # skipped threshold_check and did the maths itself
        "threshold-over": [call("database_query", account_id="ACC-102"), answer("Yes, 25,000 exceeds 10,000.")],
        # opened a ticket when none was needed (and the harness approves it)
        "check-no-ticket-needed": [call("database_query", account_id="ACC-101"),
                                   call("ticket_creation", title="Review", priority="low", description="Just in case"),
                                   answer("Opened a ticket; balance is not over.")],
        # called a wrong tool first, then the right one
        "kyc-status": [call("database_query", account_id="ACC-103"), call("crm_get_customer", account_id="ACC-103"),
                       answer("KYC is expired.")],
        # correct tool, but the answer misses the fact
        "ticket-status": [call("status_lookup", ticket_id="TKT-102"), answer("I checked the ticket.")],
    }
    report = run([by_id[k] for k in bad], bad)
    res = {r["id"]: r for r in report["results"]}
    assert res["threshold-over"]["failure"] == "missed_tool"
    assert res["check-no-ticket-needed"]["failure"] == "forbidden_action"
    assert res["kyc-status"]["completed"] and res["kyc-status"]["tool_precision"] == 0.5
    assert res["kyc-status"]["step_efficiency"] == 0.5
    assert res["ticket-status"]["failure"] == "wrong_or_incomplete_answer"


def test_a_crashing_scenario_is_recorded_and_the_exam_continues():
    by_id = {s["id"]: s for s in load_scenarios()}

    class Broken:
        async def chat(self, messages, tools):
            raise RuntimeError("an error was encountered while running the model: unexpected EOF")

    config = copy.deepcopy(load_config())
    saved = []
    scripts = {"kyc-status": REFERENCE["kyc-status"]}
    make = lambda scn: Broken() if scn["id"] == "balance-lookup" else ScriptedLLM(copy.deepcopy(scripts[scn["id"]]))  # noqa: E731
    report = asyncio.run(evaluate([by_id["balance-lookup"], by_id["kyc-status"]], make, config, load_prompt(),
                                  progress=lambda _: None, on_result=saved.append))
    res = {r["id"]: r for r in report["results"]}
    assert res["balance-lookup"]["failure"] == "error" and "unexpected EOF" in res["balance-lookup"]["error"]
    assert res["kyc-status"]["completed"]
    assert len(saved) == 2  # progress saved after each scenario
