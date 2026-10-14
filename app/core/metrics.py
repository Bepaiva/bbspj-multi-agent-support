"""Prometheus metrics exposed at GET /metrics."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "chat_requests_total", "Chat requests processed", ["route", "status"]
)
REQUEST_LATENCY = Histogram(
    "chat_request_latency_seconds",
    "End-to-end latency of /chat",
    ["route"],
    buckets=(0.25, 0.5, 1, 2, 4, 8, 15, 30, 60),
)
NODE_LATENCY = Histogram(
    "graph_node_latency_seconds",
    "Latency per orchestration node",
    ["node"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15),
)
TOOL_CALLS = Counter("agent_tool_calls_total", "Tool invocations", ["agent", "tool", "status"])
GUARDRAIL_EVENTS = Counter("guardrail_events_total", "Guardrail detections", ["stage", "flag"])
ESCALATIONS = Counter("escalations_total", "Human handoffs created", ["reason"])
RETRIEVAL_HITS = Histogram(
    "rag_retrieved_chunks", "Chunks above the relevance threshold", buckets=(0, 1, 2, 3, 5, 8)
)
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens consumed", ["agent", "kind"])
