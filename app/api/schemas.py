from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, examples=["What's the difference between the Get Clássica and the Get Smart?"])
    user_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{3,64}$", examples=["cliente1988"])
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,64}$",
                                   description="Optional conversation id; defaults to one conversation per user")


class Source(BaseModel):
    title: str
    url: str
    kind: str
    score: float | None = None


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any]
    status: str
    latency_ms: int
    summary: str = ""


class Handoff(BaseModel):
    handoff_id: str
    reason: str
    priority: str
    status: str


class Guardrails(BaseModel):
    flags: list[str]
    blocked: bool
    reason: str | None = None


class Routing(BaseModel):
    confidence: float
    reasoning: str
    fallback: bool


class ChatResponse(BaseModel):
    response: str = Field(description="Answer to show to the user")
    user_id: str
    session_id: str
    trace_id: str
    route: str | None = Field(description="knowledge | support | escalation | smalltalk | out_of_scope | blocked")
    agent: str | None = Field(description="Agent that produced the answer")
    language: str | None
    confidence: float
    sources: list[Source]
    tool_calls: list[ToolCall]
    escalated: bool
    handoff: Handoff | None
    ticket: dict[str, Any] | None
    guardrails: Guardrails
    routing: Routing
    path: list[str]
    timings_ms: dict[str, int]
