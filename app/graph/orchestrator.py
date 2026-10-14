"""LangGraph orchestration of the multi-agent system.

    START
      │
      ▼
 input_guardrail ──blocked──────────────────────────────┐
      │                                                 │
      ▼                                                 │
   router ──smalltalk / out_of_scope (direct reply)─────┤
      │                                                 │
      ├──knowledge──► knowledge_agent ──┐               │
      ├──support────► support_agent ────┤               │
      │                                 ├─needs human─► escalation_agent
      └──escalation────────────────────────────────────►│
                                        │               │
                                        ▼               ▼
                                     finalize (output guardrail, memory) ──► END
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.base import AgentContext, AgentResult
from app.agents.router import RouterAgent
from app.agents.specialists import EscalationAgent, KnowledgeAgent, SupportAgent
from app.core.config import Settings, get_settings
from app.core.logging import current_trace_id, get_logger, new_trace_id
from app.core.metrics import ESCALATIONS, GUARDRAIL_EVENTS, NODE_LATENCY
from app.graph.state import SupportState
from app.guardrails.checks import BLOCK_MESSAGES, PII_NOTICE, check_input, check_output, guess_language
from app.rag.store import KnowledgeStore
from app.services.customer_repository import CustomerRepository
from app.services.handoff import TicketStore

log = get_logger(__name__)

NodeFn = Callable[[SupportState], dict[str, Any]]


def timed(name: str) -> Callable[[NodeFn], NodeFn]:
    """Record latency + execution path for every node (metrics, logs and API response)."""

    def deco(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        def wrapper(self, state: SupportState) -> dict[str, Any]:
            start = time.perf_counter()
            update = fn(self, state)
            ms = int((time.perf_counter() - start) * 1000)
            NODE_LATENCY.labels(name).observe(ms / 1000)
            base_timings = {} if name == "input_guardrail" else dict(state.get("timings_ms") or {})
            base_path = [] if name == "input_guardrail" else list(state.get("path") or [])
            update["timings_ms"] = {**base_timings, **update.get("timings_ms", {}), name: ms}
            update["path"] = [*base_path, name]
            log.info("node.completed", node=name, latency_ms=ms)
            return update

        return wrapper

    return deco


class SupportOrchestrator:
    def __init__(self, *, llm: BaseChatModel, store: KnowledgeStore, repo: CustomerRepository,
                 tickets: TicketStore, router_llm: BaseChatModel | None = None,
                 checkpointer: BaseCheckpointSaver | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.repo = repo
        self.router = RouterAgent(router_llm or llm)
        self.knowledge = KnowledgeAgent(llm, store, self.settings.max_agent_iterations)
        self.support = SupportAgent(llm, store, repo, tickets, self.settings.max_agent_iterations + 1)
        self.escalation = EscalationAgent(llm, tickets)
        self.graph = self._build(checkpointer or InMemorySaver())

    # ------------------------------------------------------------------ graph
    def _build(self, checkpointer: BaseCheckpointSaver):
        g = StateGraph(SupportState)
        g.add_node("input_guardrail", self.input_guardrail)
        g.add_node("router", self.route)
        g.add_node("knowledge_agent", self.run_knowledge)
        g.add_node("support_agent", self.run_support)
        g.add_node("escalation_agent", self.run_escalation)
        g.add_node("finalize", self.finalize)

        g.add_edge(START, "input_guardrail")
        g.add_conditional_edges("input_guardrail", lambda s: "finalize" if s.get("blocked") else "router",
                                ["finalize", "router"])
        g.add_conditional_edges("router", self._after_router,
                                ["knowledge_agent", "support_agent", "escalation_agent", "finalize"])
        for agent_node in ("knowledge_agent", "support_agent"):
            g.add_conditional_edges(agent_node, lambda s: "escalation_agent" if s.get("needs_escalation") else "finalize",
                                    ["escalation_agent", "finalize"])
        g.add_edge("escalation_agent", "finalize")
        g.add_edge("finalize", END)
        return g.compile(checkpointer=checkpointer)

    @staticmethod
    def _after_router(state: SupportState) -> str:
        return {"knowledge": "knowledge_agent", "support": "support_agent",
                "escalation": "escalation_agent"}.get(state["route"], "finalize")

    def _history(self, state: SupportState) -> list[BaseMessage]:
        return list(state.get("messages", []))[-(self.settings.history_turns * 2):]

    def _ctx(self, state: SupportState) -> AgentContext:
        return AgentContext(user_id=state["user_id"], trace_id=state.get("trace_id"),
                            language=state.get("language", "en"))

    # ------------------------------------------------------------------ nodes
    @timed("input_guardrail")
    def input_guardrail(self, state: SupportState) -> dict[str, Any]:
        raw = state["raw_message"]
        result = check_input(raw, state["user_id"])
        for flag in result.flags:
            GUARDRAIL_EVENTS.labels("input", flag).inc()
        language = guess_language(result.text or raw)
        update: dict[str, Any] = {
            # reset per-turn fields (state persists across turns through the checkpointer)
            "user_message": result.text, "guardrail_flags": result.flags, "blocked": result.blocked,
            "block_reason": result.block_reason, "pii_notice": result.user_notice == "pii",
            "language": language, "route": "", "route_confidence": 0.0, "route_reasoning": "",
            "router_fallback": False, "standalone_query": "", "agent": "", "answer": "", "confidence": 0.0,
            "sources": [], "tool_calls": [], "needs_escalation": False, "escalation_reason": None,
            "escalation_priority": "normal", "handoff": None, "ticket": None,
        }
        if result.blocked:
            log.warning("guardrail.blocked", reason=result.block_reason, flags=result.flags)
            update.update(route="blocked", agent="guardrail", confidence=1.0,
                          answer=BLOCK_MESSAGES[result.block_reason][language if language in ("pt", "en") else "en"])
        else:
            update["messages"] = [HumanMessage(result.text)]
        return update

    @timed("router")
    def route(self, state: SupportState) -> dict[str, Any]:
        decision, fallback = self.router.route(self._history(state))
        language = decision.language if decision.language in ("pt", "en") else state.get("language", "en")
        route = decision.route
        # The router may be overconfident about off-topic requests; below the threshold we prefer a
        # specialist (which can still decline) over refusing a legitimate question.
        if route == "out_of_scope" and decision.confidence < self.settings.router_confidence_threshold:
            route = "knowledge"
        update: dict[str, Any] = {
            "route": route, "route_confidence": decision.confidence, "route_reasoning": decision.reasoning,
            "router_fallback": fallback, "language": language,
            "standalone_query": decision.standalone_query or state["user_message"],
        }
        if route == "escalation":
            update.update(needs_escalation=True, escalation_reason=f"user_request: {decision.reasoning}",
                          escalation_priority="high" if any(w in state["user_message"].lower() for w in
                                                            ("fraud", "fraude", "golpe", "roub", "stolen", "unauthori", "não autoriz"))
                          else "normal")
        if route in ("smalltalk", "out_of_scope"):
            update.update(agent="router", confidence=decision.confidence,
                          answer=decision.direct_reply or self._default_reply(route, language))
        log.info("router.decision", route=route, confidence=decision.confidence, fallback=fallback,
                 language=language)
        return update

    def _run_specialist(self, state: SupportState, agent, name: str) -> dict[str, Any]:
        ctx = self._ctx(state)
        history = self._history(state)
        # Give the specialist the router's self-contained rewrite of the request as the last turn.
        if state.get("standalone_query") and state["standalone_query"] != state["user_message"]:
            history = [*history[:-1], HumanMessage(state["standalone_query"])]
        try:
            result: AgentResult = agent.run(history, ctx)
        except Exception as exc:  # LLM outage, timeout, etc. -> graceful handoff instead of a 500
            log.error("agent.failed", agent=name, error=str(exc))
            return {"agent": name, "answer": "", "confidence": 0.0, "needs_escalation": True,
                    "escalation_reason": f"{name}_agent_error: {type(exc).__name__}", "sources": [], "tool_calls": []}

        confidence = self._confidence(name, result, ctx)
        needs_escalation = ctx.escalation_requested
        reason = ctx.escalation_reason
        if not needs_escalation and (not result.answer or confidence < 0.25):
            needs_escalation, reason = True, f"low_confidence ({confidence:.2f})"
        return {
            "agent": name, "answer": result.answer, "confidence": confidence, "sources": ctx.sources,
            "tool_calls": [t.as_dict() for t in result.tool_calls], "ticket": ctx.ticket,
            "needs_escalation": needs_escalation, "escalation_reason": reason,
            "escalation_priority": ctx.escalation_priority,
        }

    @staticmethod
    def _confidence(name: str, result: AgentResult, ctx: AgentContext) -> float:
        """Heuristic answer confidence from evidence quality (logged and returned to the client)."""
        if result.error == "max_iterations":
            return 0.3
        if name == "knowledge":
            best = max(ctx.retrieval_scores, default=0.0)
            if best >= 0.3:
                return round(min(0.95, 0.45 + best * 0.6), 2)
            if ctx.used_web_search and ctx.sources:
                return 0.7
            return 0.3 if result.answer else 0.0
        # support
        if not result.tool_calls:
            return 0.4
        errors = sum(1 for t in result.tool_calls if t.status != "ok")
        return round(max(0.3, 0.9 - 0.2 * errors), 2)

    @timed("knowledge_agent")
    def run_knowledge(self, state: SupportState) -> dict[str, Any]:
        return self._run_specialist(state, self.knowledge, "knowledge")

    @timed("support_agent")
    def run_support(self, state: SupportState) -> dict[str, Any]:
        return self._run_specialist(state, self.support, "support")

    @timed("escalation_agent")
    def run_escalation(self, state: SupportState) -> dict[str, Any]:
        ctx = self._ctx(state)
        reason = state.get("escalation_reason") or "unspecified"
        priority = state.get("escalation_priority", "normal")
        ESCALATIONS.labels(reason.split(":")[0].split(" ")[0]).inc()
        out = self.escalation.run(self._history(state), ctx, reason, priority)
        prior = (state.get("answer") or "").strip()
        answer = f"{prior}\n\n{out['message']}" if prior else out["message"]
        return {"handoff": out["handoff"], "answer": answer,
                "agent": state.get("agent") or "escalation", "confidence": state.get("confidence") or 1.0}

    @timed("finalize")
    def finalize(self, state: SupportState) -> dict[str, Any]:
        out = check_output(state.get("answer", ""))
        for flag in out.flags:
            GUARDRAIL_EVENTS.labels("output", flag).inc()
        answer = out.text or self._default_reply("error", state.get("language", "en"))
        if state.get("pii_notice"):
            answer = f"{PII_NOTICE.get(state.get('language', 'en'), PII_NOTICE['en'])}\n\n{answer}"
        update: dict[str, Any] = {"answer": answer,
                                  "guardrail_flags": [*state.get("guardrail_flags", []), *out.flags]}
        if not state.get("blocked"):
            update["messages"] = [AIMessage(answer, name=state.get("agent") or "assistant")]
        return update

    @staticmethod
    def _default_reply(kind: str, language: str) -> str:
        pt = language == "pt"
        if kind == "smalltalk":
            return "Olá! Sou o assistente da Getnet. Como posso ajudar?" if pt else "Hi! I'm Getnet's assistant. How can I help?"
        if kind == "out_of_scope":
            return ("Posso ajudar com produtos Getnet, suas vendas e recebimentos, sua maquininha e informações gerais "
                    "para o seu negócio." if pt else
                    "I can help with Getnet products, your sales and deposits, your card machine and general business information.")
        return ("Desculpe, não consegui processar sua mensagem agora. Tente novamente ou ligue 4002-4000." if pt
                else "Sorry, I couldn't process your message right now. Please try again or call 4002-4000.")

    # ------------------------------------------------------------------ public API
    def invoke(self, message: str, user_id: str, session_id: str | None = None) -> dict[str, Any]:
        trace_id = current_trace_id() or new_trace_id()
        thread_id = f"{user_id}:{session_id or 'default'}"
        start = time.perf_counter()
        final: SupportState = self.graph.invoke(
            {"raw_message": message, "user_id": user_id, "trace_id": trace_id},
            config={"configurable": {"thread_id": thread_id},
                    "metadata": {"user_id": user_id, "trace_id": trace_id},
                    "run_name": "getnet-support", "tags": ["getnet", "multi-agent"]},
        )
        total_ms = int((time.perf_counter() - start) * 1000)
        handoff = final.get("handoff")
        return {
            "response": final.get("answer", ""),
            "user_id": user_id,
            "session_id": session_id or "default",
            "trace_id": trace_id,
            "route": final.get("route"),
            "agent": final.get("agent"),
            "language": final.get("language"),
            "confidence": final.get("confidence", 0.0),
            "sources": final.get("sources", []),
            "tool_calls": final.get("tool_calls", []),
            "escalated": bool(handoff),
            "handoff": {"handoff_id": handoff["handoff_id"], "reason": handoff["reason"],
                        "priority": handoff["priority"], "status": handoff["status"]} if handoff else None,
            "ticket": {"ticket_id": final["ticket"]["ticket_id"]} if final.get("ticket") else None,
            "guardrails": {"flags": final.get("guardrail_flags", []), "blocked": bool(final.get("blocked")),
                           "reason": final.get("block_reason")},
            "routing": {"confidence": final.get("route_confidence", 0.0), "reasoning": final.get("route_reasoning", ""),
                        "fallback": final.get("router_fallback", False)},
            "path": final.get("path", []),
            "timings_ms": {**final.get("timings_ms", {}), "total": total_ms},
        }
