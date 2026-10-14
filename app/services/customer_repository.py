"""Read-only access to merchant data.

In production this would be a client for internal APIs (merchant registry, settlement
ledger, terminal management). Here it is backed by ``data/customers.json`` so the demo is
self-contained. The interface is what the Customer Support agent's tools depend on.
"""

from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.business_days import add_business_days


class CustomerNotFoundError(LookupError):
    pass


class CustomerRepository:
    def __init__(self, customers_file: Path, decline_codes_file: Path, today: date | None = None):
        self._raw = json.loads(customers_file.read_text(encoding="utf-8"))["customers"]
        self._decline_codes = json.loads(decline_codes_file.read_text(encoding="utf-8"))["codes"]
        self._today_override = today

    @property
    def today(self) -> date:
        return self._today_override or date.today()

    # ------------------------------------------------------------------ customers
    def exists(self, user_id: str) -> bool:
        return user_id in self._raw

    def get_customer(self, user_id: str) -> dict[str, Any]:
        if user_id not in self._raw:
            raise CustomerNotFoundError(user_id)
        customer = copy.deepcopy(self._raw[user_id])
        for tx in customer.get("transactions", []):
            tx["date"] = (self.today - timedelta(days=tx.pop("days_ago"))).isoformat()
        for ticket in customer.get("open_tickets", []):
            ticket["opened_on"] = (self.today - timedelta(days=ticket.pop("opened_days_ago"))).isoformat()
        return customer

    def get_profile(self, user_id: str) -> dict[str, Any]:
        c = self.get_customer(user_id)
        keys = (
            "user_id", "merchant_name", "document_type", "document_masked", "segment", "city",
            "since", "receiving_plan", "receiving_account", "enabled_products", "fees",
        )
        profile = {k: c[k] for k in keys if k in c}
        profile["terminal_count"] = len(c.get("terminals", []))
        profile["open_tickets"] = c.get("open_tickets", [])
        return profile

    def get_terminals(self, user_id: str) -> list[dict[str, Any]]:
        return self.get_customer(user_id).get("terminals", [])

    def get_transactions(
        self, user_id: str, *, status: str | None = None, date_from: date | None = None,
        date_to: date | None = None, limit: int = 20,
    ) -> list[dict[str, Any]]:
        txs = self.get_customer(user_id).get("transactions", [])
        out = []
        for tx in txs:
            d = date.fromisoformat(tx["date"])
            if status and tx["status"] != status:
                continue
            if date_from and d < date_from:
                continue
            if date_to and d > date_to:
                continue
            out.append(tx)
        out.sort(key=lambda t: (t["date"], t["time"]), reverse=True)
        return out[:limit]

    # ------------------------------------------------------------------ settlement
    def settlement_for_transaction(self, tx: dict[str, Any], plan: dict[str, Any], fees: dict[str, Any]) -> list[dict[str, Any]]:
        """Expected payout(s) for an approved transaction under the merchant's plan."""
        sale_date = date.fromisoformat(tx["date"])
        method, amount, n = tx["method"], float(tx["amount"]), int(tx.get("installments", 1))

        if method == "pix":
            fee = fees.get("pix", 0.0)
            return [{"installment": 1, "expected_date": sale_date.isoformat(), "gross": amount,
                     "fee_pct": fee, "net": round(amount * (1 - fee / 100), 2), "note": "Pix is credited instantly"}]
        if method == "debit":
            fee = fees["debit"]
            return [{"installment": 1, "expected_date": add_business_days(sale_date, plan["debit_days"]).isoformat(),
                     "gross": amount, "fee_pct": fee, "net": round(amount * (1 - fee / 100), 2)}]
        # credit
        if n == 1:
            fee = fees["credit_1x"]
            return [{"installment": 1, "expected_date": add_business_days(sale_date, plan["credit_days"]).isoformat(),
                     "gross": amount, "fee_pct": fee, "net": round(amount * (1 - fee / 100), 2)}]
        fee = fees["credit_installments_from"]
        if plan.get("installments_mode") == "emissor":
            return [{"installment": "full", "expected_date": add_business_days(sale_date, plan["credit_days"]).isoformat(),
                     "gross": amount, "fee_pct": fee, "net": round(amount * (1 - fee / 100), 2),
                     "note": "Installments financed by the issuer - full amount at once"}]
        per = round(amount / n, 2)
        payouts = []
        for i in range(1, n + 1):
            due = sale_date + timedelta(days=31 + 30 * (i - 1))
            payouts.append({"installment": f"{i}/{n}", "expected_date": due.isoformat(), "gross": per,
                            "fee_pct": fee, "net": round(per * (1 - fee / 100), 2)})
        payouts[0]["note"] = "Merchant-financed installments (parcelado lojista): D+31, then every 30 days - can be advanced (antecipação)"
        return payouts

    def get_settlement_schedule(self, user_id: str, sale_date: date | None = None) -> dict[str, Any]:
        customer = self.get_customer(user_id)
        plan, fees = customer["receiving_plan"], customer["fees"]
        sale_date = sale_date or (self.today - timedelta(days=1))
        approved = [t for t in customer["transactions"] if t["status"] == "approved" and t["date"] == sale_date.isoformat()]
        items, total_gross, total_net = [], 0.0, 0.0
        for tx in approved:
            payouts = self.settlement_for_transaction(tx, plan, fees)
            total_gross += tx["amount"]
            total_net += sum(p["net"] for p in payouts)
            items.append({"transaction_id": tx["id"], "time": tx["time"], "method": tx["method"], "brand": tx["brand"],
                          "amount": tx["amount"], "installments": tx.get("installments", 1), "payouts": payouts})
        return {
            "sale_date": sale_date.isoformat(),
            "today": self.today.isoformat(),
            "receiving_plan": plan["name"],
            "receiving_account": customer["receiving_account"]["masked"],
            "automatic_advance": plan.get("automatic_advance", False),
            "approved_sales": len(items),
            "total_gross": round(total_gross, 2),
            "total_net_expected": round(total_net, 2),
            "items": items,
        }

    # ------------------------------------------------------------------ decline codes
    def explain_decline_code(self, code: str) -> dict[str, Any] | None:
        info = self._decline_codes.get(code.strip().upper())
        return {"code": code, **info} if info else None


@lru_cache
def get_customer_repository() -> CustomerRepository:
    s = get_settings()
    return CustomerRepository(s.customers_file, s.decline_codes_file)
