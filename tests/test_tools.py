"""The back-end tool contracts: valid input works, invalid input is rejected with a clear error."""

import tools


def test_nine_tools_and_two_state_changing():
    assert len(tools.TOOLS) == 9
    assert tools.STATE_CHANGING_TOOLS == {"email_draft", "ticket_creation"}


def test_database_query_found_and_not_found():
    ok = tools.database_query({"account_id": "acc-102"})
    assert ok["found"] and ok["balance_eur"] == 25000.0
    missing = tools.database_query({"account_id": "ACC-404"})
    assert missing["found"] is False and missing["status"] == "not_found"


def test_invalid_arguments_rejected():
    bad = tools.ticket_creation({"title": "Test", "priority": "urgent", "description": "Bad priority"})
    assert bad["error_type"] == "validation_error" and "priority" in bad["error"]
    bad_date = tools.calendar_lookup({"calendar_name": "compliance", "date_iso": "next friday"})
    assert bad_date["error_type"] == "validation_error"


def test_threshold_check_is_exact():
    res = tools.threshold_check({"amount": 25000, "threshold": 10000})
    assert res["exceeds_threshold"] is True and res["difference"] == 15000.0
    assert tools.threshold_check({"amount": 10000, "threshold": 10000})["exceeds_threshold"] is False


def test_document_search_filters_by_category():
    res = tools.document_search({"query": "KYC expired", "category": "compliance"})
    assert "KYC_Policy.md" in res["sources"]
    assert tools.document_search({"query": "KYC expired", "category": "hr"})["matches"] == []
