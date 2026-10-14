"""Support tickets and human-handoff queue.

Tickets/handoffs are appended to a JSONL file (a stand-in for a real ticketing system such
as Zendesk/ServiceNow) and, if ``HANDOFF_WEBHOOK_URL`` is configured, posted to a chat
channel (e.g. a Microsoft Teams or Slack incoming webhook) so a human operator is paged.
"""

from __future__ import annotations

import json
import secrets
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
_lock = threading.Lock()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC):%Y%m%d}-{secrets.token_hex(3).upper()}"


class TicketStore:
    def __init__(self, path: Path, webhook_url: str | None = None) -> None:
        self.path = path
        self.webhook_url = webhook_url
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict[str, Any]) -> None:
        with _lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def create_ticket(self, *, user_id: str, category: str, description: str, priority: str = "normal",
                      trace_id: str | None = None) -> dict[str, Any]:
        record = {
            "type": "support_ticket", "ticket_id": _new_id("SUP"), "user_id": user_id, "category": category,
            "description": description, "priority": priority, "status": "open",
            "created_at": datetime.now(UTC).isoformat(), "trace_id": trace_id,
        }
        self._append(record)
        log.info("ticket.created", ticket_id=record["ticket_id"], category=category, priority=priority)
        return record

    def create_handoff(self, *, user_id: str, reason: str, summary: str, last_user_message: str,
                       priority: str = "normal", trace_id: str | None = None) -> dict[str, Any]:
        record = {
            "type": "human_handoff", "handoff_id": _new_id("HND"), "user_id": user_id, "reason": reason,
            "priority": priority, "summary": summary, "last_user_message": last_user_message,
            "status": "waiting_agent", "queue": "getnet-merchant-support",
            "created_at": datetime.now(UTC).isoformat(), "trace_id": trace_id,
        }
        self._append(record)
        self._notify(record)
        log.info("handoff.created", handoff_id=record["handoff_id"], reason=reason, priority=priority)
        return record

    def _notify(self, record: dict[str, Any]) -> None:
        if not self.webhook_url:
            return
        text = (f"🚨 Human handoff {record['handoff_id']} ({record['priority']})\n"
                f"Merchant: {record['user_id']} | Reason: {record['reason']}\n{record['summary']}")
        try:
            httpx.post(self.webhook_url, json={"text": text}, timeout=5)
        except httpx.HTTPError as exc:  # never fail the user request because paging failed
            log.warning("handoff.webhook_failed", error=str(exc))


_store: TicketStore | None = None


def get_ticket_store() -> TicketStore:
    global _store
    if _store is None:
        s = get_settings()
        _store = TicketStore(s.handoff_file, s.handoff_webhook_url)
    return _store
