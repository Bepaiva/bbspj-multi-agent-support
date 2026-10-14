"""Tools available to the Knowledge agent (and the KB tool to the Support agent)."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool

from app.agents.base import AgentContext
from app.core.metrics import RETRIEVAL_HITS
from app.rag.store import KnowledgeStore
from app.tools.web_search import WebSearchError, web_search


def make_kb_tool(store: KnowledgeStore, ctx: AgentContext) -> BaseTool:
    @tool
    def search_getnet_knowledge_base(query: str) -> dict[str, Any]:
        """Search Getnet's official knowledge base (products, fees, Pix, payment link, crediário,
        antecipação, receiving terms, card machines, troubleshooting, support channels).
        Use it for ANY question about Getnet. Write the query as a short, specific search
        phrase; Portuguese product names (e.g. 'antecipação', 'maquininha') work well."""
        hits = store.search(query)
        RETRIEVAL_HITS.observe(len(hits))
        results = []
        for doc, score in hits:
            ctx.retrieval_scores.append(score)
            ctx.add_source(doc.metadata.get("title", ""), doc.metadata.get("source", ""), "knowledge_base", score)
            results.append({"content": doc.page_content, "source": doc.metadata.get("source"),
                            "relevance": round(score, 3)})
        if not results:
            return {"results": [], "note": "No relevant passages found in the knowledge base."}
        return {"results": results}

    return search_getnet_knowledge_base


def make_web_search_tool(ctx: AgentContext) -> BaseTool:
    @tool
    def search_web(query: str) -> dict[str, Any]:
        """Search the public web for general-purpose or real-time information that is NOT in the
        Getnet knowledge base: weather forecasts, currency exchange rates, news, public facts.
        Include the place and the date when relevant (e.g. 'weather forecast Porto Alegre tomorrow')."""
        try:
            results = web_search(query)
        except WebSearchError as exc:
            return {"results": [], "error": f"web search unavailable: {exc}"}
        ctx.used_web_search = True
        for r in results:
            ctx.add_source(r["title"], r["url"], "web")
        return {"results": results}

    return search_web


def make_escalation_tool(ctx: AgentContext) -> BaseTool:
    @tool
    def request_human_agent(reason: str, priority: str = "normal") -> str:
        """Hand the conversation over to a human Getnet support operator. Use ONLY when: the user
        explicitly asks for a human; the issue needs an action you cannot perform (e.g. cancel a
        sale from a previous day, change bank account, dispute/chargeback, suspected fraud or
        account takeover); or you could not resolve the problem with the available tools.
        priority: 'normal' or 'high' (high = money at risk, fraud, business fully stopped)."""
        ctx.escalation_requested = True
        ctx.escalation_reason = reason
        ctx.escalation_priority = "high" if priority == "high" else "normal"
        return "Escalation registered. Tell the user a human specialist will take over; do not promise a time."

    return request_human_agent
