"""Shared state passed between the orchestration nodes."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SupportState(TypedDict, total=False):
    # Conversation memory (persisted per thread by the checkpointer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Request
    user_id: str
    trace_id: str
    raw_message: str
    user_message: str  # sanitized (PII redacted)

    # Guardrails
    guardrail_flags: list[str]
    blocked: bool
    block_reason: str | None
    pii_notice: bool

    # Routing
    language: str
    route: str
    route_confidence: float
    route_reasoning: str
    router_fallback: bool
    standalone_query: str

    # Agent output
    agent: str
    answer: str
    confidence: float
    sources: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]

    # Escalation
    needs_escalation: bool
    escalation_reason: str | None
    escalation_priority: str
    handoff: dict[str, Any] | None
    ticket: dict[str, Any] | None

    # Observability
    timings_ms: dict[str, int]
    path: list[str]
