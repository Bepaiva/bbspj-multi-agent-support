"""General-purpose web search used by the Knowledge agent (weather, exchange rates, news...).

Providers: Tavily (if TAVILY_API_KEY is set; better quality, made for LLM agents) or
DuckDuckGo (no key required). Results are normalised to {title, url, snippet}.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


class WebSearchError(RuntimeError):
    pass


def web_search(query: str, max_results: int | None = None, settings: Settings | None = None) -> list[dict[str, Any]]:
    s = settings or get_settings()
    n = max_results or s.web_search_max_results
    provider = s.web_search_provider
    if provider == "tavily" and s.tavily_api_key:
        return _tavily(query, n, s.tavily_api_key)
    if provider == "none":
        raise WebSearchError("web search disabled")
    return _duckduckgo(query, n)


def _tavily(query: str, n: int, api_key: str) -> list[dict[str, Any]]:
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": n, "include_answer": False},
            timeout=20,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise WebSearchError(str(exc)) from exc
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:700]}
            for r in resp.json().get("results", [])]


def _duckduckgo(query: str, n: int) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=n))
    except Exception as exc:  # ddgs raises a variety of exceptions (rate limit, timeout)
        raise WebSearchError(str(exc)) from exc
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")[:700]} for r in raw]
