import json

import pytest

from app.agents.base import AgentContext
from app.services.customer_repository import CustomerNotFoundError
from app.tools.support_tools import make_support_tools


def _tools(repo, tickets, user_id="cliente1988"):
    ctx = AgentContext(user_id=user_id, trace_id="t-1")
    return {t.name: t for t in make_support_tools(repo, tickets, ctx)}, ctx


def test_settlement_schedule_for_yesterday(repo):
    sched = repo.get_settlement_schedule("cliente1988")
    assert sched["sale_date"] == "2026-10-13"
    assert sched["approved_sales"] == 5  # the declined sale is excluded
    by_id = {i["transaction_id"]: i for i in sched["items"]}
    assert by_id["TX-90001"]["payouts"][0]["expected_date"] == "2026-10-14"  # debit D+1
    assert by_id["TX-90002"]["payouts"][0]["expected_date"] == "2026-10-15"  # credit D+2
    assert by_id["TX-90003"]["payouts"][0]["expected_date"] == "2026-10-13"  # pix instant
    installments = by_id["TX-90004"]["payouts"]
    assert len(installments) == 3 and installments[0]["expected_date"] == "2026-11-13"  # D+31
    assert sched["total_net_expected"] < sched["total_gross"]


def test_unknown_customer_raises(repo):
    with pytest.raises(CustomerNotFoundError):
        repo.get_customer("nobody")


def test_tools_are_scoped_to_the_authenticated_user(repo, tickets):
    tools, _ = _tools(repo, tickets, "cliente2024")
    profile = tools["get_merchant_profile"].invoke({})
    assert profile["user_id"] == "cliente2024"
    # No tool accepts a user id argument - the LLM cannot ask for another merchant.
    for t in tools.values():
        assert "user_id" not in t.args


def test_tools_handle_unknown_merchant_gracefully(repo, tickets):
    tools, _ = _tools(repo, tickets, "ghost999")
    assert tools["get_settlement_schedule"].invoke({})["error"] == "merchant_not_found"


def test_terminal_diagnostics_shows_offline_classica(repo, tickets):
    tools, _ = _tools(repo, tickets)
    terminals = tools["get_terminal_diagnostics"].invoke({})["terminals"]
    offline = [t for t in terminals if t["status"] == "offline"]
    assert offline and offline[0]["model"] == "Get Clássica"


def test_declined_transactions_and_code_explanation(repo, tickets):
    tools, _ = _tools(repo, tickets)
    declined = tools["get_recent_transactions"].invoke({"status": "declined", "days": 3})["transactions"]
    assert {t["decline_code"] for t in declined} == {"05", "51", "55"}
    info = tools["explain_decline_code"].invoke({"code": "51"})
    assert "Insufficient" in info["meaning"]


def test_invalid_date_is_reported(repo, tickets):
    tools, _ = _tools(repo, tickets)
    assert tools["get_settlement_schedule"].invoke({"sale_date": "13/10/2026"})["error"] == "invalid_date"


def test_ticket_creation_is_persisted(repo, tickets):
    tools, ctx = _tools(repo, tickets)
    out = tools["create_support_ticket"].invoke({"category": "terminal", "description": "Get Clássica offline"})
    assert out["ticket_id"].startswith("SUP-") and ctx.ticket["ticket_id"] == out["ticket_id"]
    record = json.loads(tickets.path.read_text().strip().splitlines()[-1])
    assert record["user_id"] == "cliente1988" and record["trace_id"] == "t-1"
