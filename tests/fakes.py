"""Deterministic stand-ins for the LLM so the full orchestration can be tested offline.

``PolicyChatModel`` behaves like a well-behaved tool-calling model: it recognises which agent
is calling it (from the bound tools / system prompt), decides which tool to call with simple
rules, then writes a final answer from the tool results. It records every prompt it receives
so tests can assert on what reached the "LLM" (e.g. that PII was redacted).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from app.agents.router import keyword_route


def _call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}",
                                              "type": "tool_call"}])


class PolicyChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bound_tools: list[str] = []
    calls: list[list[BaseMessage]] = []
    fail_agents: set[str] = set()  # e.g. {"support"} to simulate an LLM outage for that agent
    router_override: dict[str, Any] | None = None

    @property
    def _llm_type(self) -> str:
        return "policy-fake"

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        clone = self.model_copy(update={"bound_tools": names})
        clone.calls = self.calls  # share the recorder
        return clone

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager: CallbackManagerForLLMRun | None = None,
                  **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    # ------------------------------------------------------------------ policy
    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        system = next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")
        last_human = next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        tools = set(self.bound_tools)

        if "RouteDecision" in tools:
            decision = keyword_route(last_human).model_dump()
            decision["confidence"] = 0.9
            if self.router_override:
                decision.update(self.router_override)
            return _call("RouteDecision", decision)

        if "operator_summary" in system:  # escalation agent
            return AIMessage(json.dumps({"operator_summary": f"Lojista pediu ajuda: {last_human[:80]}",
                                         "user_message": "A human specialist will continue. Protocol: PROTOCOL_PLACEHOLDER."}))

        agent = "support" if "get_settlement_schedule" in tools else "knowledge" if "search_web" in tools else "other"
        if agent in self.fail_agents:
            raise RuntimeError(f"simulated LLM outage for {agent}")

        q = last_human.lower()
        if not tool_msgs:
            if agent == "knowledge":
                if any(w in q for w in ("weather", "previsão", "exchange", "câmbio", "euro", "dollar")):
                    return _call("search_web", {"query": last_human})
                if "human" in q or "atendente" in q:
                    return _call("request_human_agent", {"reason": "user asked for a human"})
                return _call("search_getnet_knowledge_base", {"query": last_human})
            if agent == "support":
                if "human" in q or "atendente" in q or "refund" in q:
                    return _call("request_human_agent", {"reason": "needs human", "priority": "high"})
                if any(w in q for w in ("deposit", "receive", "money", "dinheiro", "cair")):
                    return _call("get_settlement_schedule", {})
                if any(w in q for w in ("connect", "internet", "conex", "wifi")):
                    return _call("get_terminal_diagnostics", {})
                if any(w in q for w in ("decline", "negad", "recusad", "error")):
                    return _call("get_recent_transactions", {"status": "declined", "days": 3})
                return _call("get_merchant_profile", {})

        # Second step for declines: explain the most recent code.
        if agent == "support" and tool_msgs and tool_msgs[-1].name == "get_recent_transactions":
            data = json.loads(str(tool_msgs[-1].content))
            if data.get("transactions"):
                return _call("explain_decline_code", {"code": data["transactions"][0]["decline_code"]})

        evidence = " | ".join(str(m.content)[:1500] for m in tool_msgs)
        return AIMessage(f"[{agent} answer] {evidence}")


class FailingChatModel(BaseChatModel):
    """Raises on every call - simulates an LLM provider outage."""

    @property
    def _llm_type(self) -> str:
        return "failing-fake"

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise TimeoutError("LLM provider timeout")
