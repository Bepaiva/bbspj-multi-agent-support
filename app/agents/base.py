"""A small, explicit tool-calling loop shared by the specialist agents.

We intentionally keep the loop in-house (instead of a prebuilt ReAct agent) so that:
* every tool call is traced (name, args, latency, status) and returned in the API response;
* iteration limits and failure handling are explicit;
* agents share a typed ``AgentContext`` that tools write to (sources, escalation requests).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.core.logging import get_logger
from app.core.metrics import LLM_TOKENS, TOOL_CALLS

log = get_logger(__name__)


@dataclass
class ToolTrace:
    tool: str
    args: dict[str, Any]
    status: str
    latency_ms: int
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "status": self.status, "latency_ms": self.latency_ms,
                "summary": self.summary}


@dataclass
class AgentContext:
    """Per-request scratchpad shared between an agent and its tools."""

    user_id: str
    trace_id: str | None = None
    language: str = "en"
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    used_web_search: bool = False
    escalation_requested: bool = False
    escalation_reason: str | None = None
    escalation_priority: str = "normal"
    ticket: dict[str, Any] | None = None

    def add_source(self, title: str, url: str, kind: str, score: float | None = None) -> None:
        if any(s["url"] == url for s in self.sources):
            return
        self.sources.append({"title": title, "url": url, "kind": kind, **({"score": round(score, 3)} if score is not None else {})})


@dataclass
class AgentResult:
    agent: str
    answer: str
    tool_calls: list[ToolTrace]
    iterations: int
    error: str | None = None


ToolFactory = Callable[[AgentContext], list[BaseTool]]


class ToolCallingAgent:
    name: str = "agent"

    def __init__(self, llm: BaseChatModel, system_prompt: str, tool_factory: ToolFactory,
                 max_iterations: int = 5) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.tool_factory = tool_factory
        self.max_iterations = max_iterations

    def _system(self, ctx: AgentContext) -> SystemMessage:
        return SystemMessage(self.system_prompt.format(
            language=ctx.language, user_id=ctx.user_id, today=date.today().strftime("%A, %Y-%m-%d")))

    def run(self, messages: list[BaseMessage], ctx: AgentContext) -> AgentResult:
        tools = self.tool_factory(ctx)
        by_name = {t.name: t for t in tools}
        llm_with_tools = self.llm.bind_tools(tools) if tools else self.llm
        convo: list[BaseMessage] = [self._system(ctx), *messages]
        traces: list[ToolTrace] = []

        for iteration in range(1, self.max_iterations + 1):
            ai: AIMessage = llm_with_tools.invoke(convo)
            self._count_tokens(ai)
            convo.append(ai)
            if not ai.tool_calls:
                return AgentResult(self.name, _text(ai), traces, iteration)
            for call in ai.tool_calls:
                convo.append(self._execute(call, by_name, traces))

        # Out of iterations: ask for a final answer without tools.
        log.warning("agent.max_iterations", agent=self.name)
        ai = self.llm.invoke([*convo, SystemMessage(
            "You have reached the tool-call limit. Answer now with the information gathered so far.")])
        self._count_tokens(ai)
        return AgentResult(self.name, _text(ai), traces, self.max_iterations, error="max_iterations")

    def _execute(self, call: dict[str, Any], by_name: dict[str, BaseTool], traces: list[ToolTrace]) -> ToolMessage:
        name, args = call["name"], call.get("args", {})
        start = time.perf_counter()
        tool = by_name.get(name)
        if tool is None:
            content, status = json.dumps({"error": f"unknown tool {name}"}), "error"
        else:
            try:
                result = tool.invoke(args)
                content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
                status = "ok"
            except Exception as exc:  # tool failures are reported back to the model, not raised
                log.warning("tool.failed", agent=self.name, tool=name, error=str(exc))
                content, status = json.dumps({"error": str(exc)}), "error"
        latency = int((time.perf_counter() - start) * 1000)
        TOOL_CALLS.labels(self.name, name, status).inc()
        traces.append(ToolTrace(name, args, status, latency, content[:160]))
        log.info("tool.called", agent=self.name, tool=name, status=status, latency_ms=latency)
        return ToolMessage(content=content, tool_call_id=call["id"], name=name)

    def _count_tokens(self, ai: AIMessage) -> None:
        usage = getattr(ai, "usage_metadata", None) or {}
        if usage:
            LLM_TOKENS.labels(self.name, "input").inc(usage.get("input_tokens", 0))
            LLM_TOKENS.labels(self.name, "output").inc(usage.get("output_tokens", 0))


def _text(ai: AIMessage) -> str:
    if isinstance(ai.content, str):
        return ai.content.strip()
    parts = [p.get("text", "") for p in ai.content if isinstance(p, dict)]
    return "".join(parts).strip()
