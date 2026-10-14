"""Router agent: classifies the request and picks the specialist (structured output).

Reliability: if the LLM call fails or returns an invalid structure, a deterministic keyword
router takes over so the system degrades gracefully instead of failing the request.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.prompts.router import ROUTER_SYSTEM_PROMPT

log = get_logger(__name__)

Route = Literal["knowledge", "support", "escalation", "smalltalk", "out_of_scope"]


class RouteDecision(BaseModel):
    """Routing decision for the latest user message."""

    route: Route = Field(description="Specialist that should handle the message")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the route, 0-1")
    language: str = Field(description="ISO 639-1 code of the user's language, e.g. 'pt' or 'en'")
    reasoning: str = Field(description="One short sentence explaining the decision")
    standalone_query: str = Field(description="Self-contained rewrite of the user's request")
    direct_reply: str = Field(default="", description="Reply text, only for smalltalk/out_of_scope")


class RouterAgent:
    name = "router"

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.structured = llm.with_structured_output(RouteDecision)

    def route(self, history: list[BaseMessage]) -> tuple[RouteDecision, bool]:
        """Return (decision, used_fallback)."""
        try:
            decision = self.structured.invoke([SystemMessage(ROUTER_SYSTEM_PROMPT), *history])
            if isinstance(decision, dict):
                decision = RouteDecision.model_validate(decision)
            if decision is None:
                raise ValueError("empty router output")
            return decision, False
        except Exception as exc:
            log.warning("router.llm_failed_using_fallback", error=str(exc))
            text = str(history[-1].content) if history else ""
            return keyword_route(text), True


# ---------------------------------------------------------------------------- fallback

def _norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


_PT_HINTS = re.compile(r"\b(o|a|os|as|um|uma|de|do|da|com|que|nao|minha|meu|como|quando|voce|para|quero|preciso|"
                       r"falar|vendas|maquininha|oi|ola|qual|quais|posso)\b")
_ESCALATION = re.compile(r"\b(human|humano|atendente|person|pessoa|operator|operador|reclama\w*|complain\w*|fraud\w*|"
                         r"golpe|roubad\w*|stolen|nao autoriz\w*|unauthori[sz]ed|didn'?t authori[sz]e|ouvidoria)\b")
# Strong signals that the user has a problem with their own operation.
_SUPPORT_STRONG = re.compile(r"\b(declin\w*|negad\w*|recusad\w*|won'?t connect|not connecting|nao conecta|sem conexao|"
                             r"deposited|depositad\w*|cai na conta|cair na conta|caiu|vendas de ontem|yesterday'?s sales|"
                             r"not working|nao funciona|error|erro|am i on|do i pay|meu plano|minhas taxas)\b")
# General "how does it work / can I / do I need" questions are product knowledge.
_GENERAL_QUESTION = re.compile(r"^\s*(do i need|can i|how (does|do|many|much)|what('s| is| are)|is it|does|"
                               r"como funciona|posso|preciso|quantas|qual a diferenca|o que e)\b")
_SUPPORT_POSSESSIVE = re.compile(r"\b(my|minha|meu|minhas|meus)\b.*\b(machine|maquin\w*|sales|vendas|money|dinheiro|"
                                 r"deposit\w*|payment\w*|pagamento\w*|account|conta|terminal|receb\w*|transac\w*)\b")
_SMALLTALK = re.compile(r"^\s*(hi|hello|hey|oi|ola|bom dia|boa tarde|boa noite|thanks|thank you|obrigad\w*|valeu|tchau|bye)\b[\s!.?]*$")


def keyword_route(text: str) -> RouteDecision:
    t = _norm(text)
    language = "pt" if _PT_HINTS.search(t) else "en"
    if _SMALLTALK.search(t):
        route: Route = "smalltalk"
        reply = ("Olá! Sou o assistente da Getnet. Como posso ajudar?" if language == "pt"
                 else "Hi! I'm Getnet's assistant. How can I help you today?")
        return RouteDecision(route=route, confidence=0.6, language=language, reasoning="keyword fallback: greeting",
                             standalone_query=text, direct_reply=reply)
    if _ESCALATION.search(t):
        route = "escalation"
    elif _SUPPORT_STRONG.search(t):
        route = "support"
    elif _GENERAL_QUESTION.search(t):
        route = "knowledge"
    elif _SUPPORT_POSSESSIVE.search(t):
        route = "support"
    else:
        route = "knowledge"
    return RouteDecision(route=route, confidence=0.5, language=language, reasoning=f"keyword fallback: {route}",
                         standalone_query=text)
