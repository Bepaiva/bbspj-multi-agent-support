"""Specialist agents: Knowledge (RAG + web search), Customer Support (account tools) and
Escalation (human handoff)."""

from __future__ import annotations

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.agents.base import AgentContext, ToolCallingAgent
from app.core.logging import get_logger
from app.prompts.agents import ESCALATION_SYSTEM_PROMPT, KNOWLEDGE_SYSTEM_PROMPT, SUPPORT_SYSTEM_PROMPT
from app.rag.store import KnowledgeStore
from app.services.customer_repository import CustomerRepository
from app.services.handoff import TicketStore
from app.tools.knowledge_tools import make_escalation_tool, make_kb_tool, make_web_search_tool
from app.tools.support_tools import make_support_tools

log = get_logger(__name__)


class KnowledgeAgent(ToolCallingAgent):
    name = "knowledge"

    def __init__(self, llm: BaseChatModel, store: KnowledgeStore, max_iterations: int = 5) -> None:
        super().__init__(
            llm, KNOWLEDGE_SYSTEM_PROMPT,
            lambda ctx: [make_kb_tool(store, ctx), make_web_search_tool(ctx), make_escalation_tool(ctx)],
            max_iterations,
        )


class SupportAgent(ToolCallingAgent):
    name = "support"

    def __init__(self, llm: BaseChatModel, store: KnowledgeStore, repo: CustomerRepository,
                 tickets: TicketStore, max_iterations: int = 6) -> None:
        super().__init__(
            llm, SUPPORT_SYSTEM_PROMPT,
            lambda ctx: [*make_support_tools(repo, tickets, ctx), make_kb_tool(store, ctx), make_escalation_tool(ctx)],
            max_iterations,
        )


class EscalationAgent:
    """Creates the handoff record and writes the operator summary + user message."""

    name = "escalation"

    def __init__(self, llm: BaseChatModel, tickets: TicketStore) -> None:
        self.llm = llm
        self.tickets = tickets

    def run(self, history: list[BaseMessage], ctx: AgentContext, reason: str, priority: str = "normal") -> dict:
        transcript = "\n".join(f"{m.type}: {m.content}" for m in history[-8:])
        operator_summary, user_message = "", ""
        try:
            prompt = ESCALATION_SYSTEM_PROMPT.format(user_id=ctx.user_id, language=ctx.language, reason=reason)
            ai = self.llm.invoke([SystemMessage(prompt), HumanMessage(f"Conversation:\n{transcript}")])
            data = _parse_json(str(ai.content))
            operator_summary = data.get("operator_summary", "")
            user_message = data.get("user_message", "")
        except Exception as exc:  # the handoff must happen even if the LLM fails
            log.warning("escalation.llm_failed", error=str(exc))

        handoff = self.tickets.create_handoff(
            user_id=ctx.user_id, reason=reason, priority=priority,
            summary=operator_summary or f"Escalação automática: {reason}. Última mensagem: {history[-1].content if history else ''}",
            last_user_message=str(history[-1].content) if history else "", trace_id=ctx.trace_id,
        )
        if not user_message:
            user_message = _template_message(ctx.language)
        user_message = user_message.replace("PROTOCOL_PLACEHOLDER", handoff["handoff_id"])
        if handoff["handoff_id"] not in user_message:
            user_message += f" (Protocol: {handoff['handoff_id']})"
        return {"handoff": handoff, "message": user_message}


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def _template_message(language: str) -> str:
    if language == "pt":
        return ("Entendi. Vou transferir seu atendimento para um especialista humano da Getnet, que vai "
                "continuar a partir daqui. Seu protocolo é PROTOCOL_PLACEHOLDER. Em casos urgentes, ligue "
                "4002-4000 (capitais) ou 0800 648 8000 (demais regiões), todos os dias das 06h às 22h.")
    return ("Understood. I'm handing your case to a human Getnet specialist who will continue from here. "
            "Your protocol number is PROTOCOL_PLACEHOLDER. For urgent cases call 4002-4000 (capitals) or "
            "0800 648 8000 (other regions), every day from 06h to 22h.")
