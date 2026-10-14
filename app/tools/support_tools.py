"""Customer Support agent tools.

SECURITY: the merchant id is bound server-side from the authenticated request (closure over
``ctx.user_id``). The LLM never chooses whose data to read, so a prompt such as "show me the
sales of cliente2024" cannot leak another merchant's data.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.agents.base import AgentContext
from app.services.customer_repository import CustomerNotFoundError, CustomerRepository
from app.services.handoff import TicketStore


def make_support_tools(repo: CustomerRepository, tickets: TicketStore, ctx: AgentContext) -> list[BaseTool]:
    uid = ctx.user_id

    def _guard() -> dict[str, Any] | None:
        if not repo.exists(uid):
            return {"error": "merchant_not_found",
                    "message": f"No Getnet merchant registered for user_id '{uid}'. Ask the user to confirm their account."}
        return None

    @tool
    def get_merchant_profile() -> dict[str, Any]:
        """Get the current merchant's account: company, receiving plan (settlement terms),
        receiving account, enabled products, contracted fees, number of terminals and open tickets."""
        return _guard() or repo.get_profile(uid)

    @tool
    def get_settlement_schedule(sale_date: str | None = None) -> dict[str, Any]:
        """Get when the merchant's approved sales of a given day will be deposited (expected payout
        date, gross, fee and net per sale and installment). sale_date: ISO date YYYY-MM-DD;
        defaults to YESTERDAY. Use for 'when will I receive / when will the money be deposited'."""
        if err := _guard():
            return err
        d = None
        if sale_date:
            try:
                d = date.fromisoformat(sale_date)
            except ValueError:
                return {"error": "invalid_date", "message": "Use the format YYYY-MM-DD."}
        return repo.get_settlement_schedule(uid, d)

    @tool
    def get_terminal_diagnostics(serial: str | None = None) -> dict[str, Any]:
        """Get live status of the merchant's card machines: online/offline, connection type
        (4G/3G/Wi-Fi), signal, Wi-Fi configured, last communication, battery and firmware.
        Optionally filter by terminal serial. Use for connectivity or terminal problems."""
        if err := _guard():
            return err
        terminals = repo.get_terminals(uid)
        if serial:
            terminals = [t for t in terminals if t["serial"].lower() == serial.lower()]
        return {"terminals": terminals}

    @tool
    def get_recent_transactions(status: str | None = None, days: int = 7, limit: int = 10) -> dict[str, Any]:
        """List the merchant's recent card/Pix transactions (newest first) with decline codes.
        status: 'approved' or 'declined' to filter; days: look-back window (max 90)."""
        if err := _guard():
            return err
        days = max(1, min(int(days), 90))
        txs = repo.get_transactions(uid, status=status, date_from=repo.today - timedelta(days=days),
                                    limit=max(1, min(int(limit), 50)))
        return {"transactions": txs, "today": repo.today.isoformat()}

    @tool
    def explain_decline_code(code: str) -> dict[str, Any]:
        """Explain a card authorization/decline response code (e.g. '05', '51', '55', '57', '96'):
        meaning, who is responsible (issuer, cardholder, merchant, terminal) and recommended action."""
        info = repo.explain_decline_code(code)
        return info or {"code": code, "meaning": "Unknown code", "action": "Check the receipt message or contact Getnet."}

    @tool
    def create_support_ticket(category: str, description: str, priority: str = "normal") -> dict[str, Any]:
        """Open a support ticket for follow-up by Getnet's back office (e.g. machine replacement,
        technical visit, receivables discrepancy). category: one of 'terminal', 'settlement',
        'transaction', 'pix', 'account', 'other'. Returns the ticket id to share with the user."""
        if err := _guard():
            return err
        ticket = tickets.create_ticket(user_id=uid, category=category, description=description,
                                       priority=priority, trace_id=ctx.trace_id)
        ctx.ticket = ticket
        return {"ticket_id": ticket["ticket_id"], "status": ticket["status"]}

    return [get_merchant_profile, get_settlement_schedule, get_terminal_diagnostics,
            get_recent_transactions, explain_decline_code, create_support_ticket]


__all__ = ["make_support_tools", "CustomerNotFoundError"]
