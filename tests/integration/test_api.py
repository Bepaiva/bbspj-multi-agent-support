import pytest
from fastapi.testclient import TestClient

import app.main as main


@pytest.fixture
def client(orchestrator):
    main._state["orchestrator"] = orchestrator
    with TestClient(main.app) as c:
        yield c
    main._state["orchestrator"] = None


def test_chat_endpoint_returns_structured_json(client):
    resp = client.post("/chat", json={"message": "Can I sell through WhatsApp using the Payment Link?",
                                      "user_id": "cliente1988"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "knowledge" and body["response"]
    assert {"sources", "tool_calls", "trace_id", "timings_ms", "guardrails", "escalated"} <= body.keys()
    assert resp.headers["x-trace-id"]


def test_request_validation(client):
    assert client.post("/chat", json={"message": "", "user_id": "cliente1988"}).status_code == 422
    assert client.post("/chat", json={"message": "hi", "user_id": "bad id!"}).status_code == 422
    assert client.post("/chat", json={"message": "hi"}).status_code == 422


def test_trace_id_is_propagated(client):
    resp = client.post("/chat", json={"message": "Oi", "user_id": "cliente1988"}, headers={"x-trace-id": "abc123"})
    assert resp.headers["x-trace-id"] == "abc123"
    assert resp.json()["trace_id"] == "abc123"


def test_health_and_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    client.post("/chat", json={"message": "Oi", "user_id": "cliente1988"})
    metrics = client.get("/metrics").text
    assert "chat_requests_total" in metrics and "graph_node_latency_seconds" in metrics


def test_missing_api_key_returns_503():
    main._state["orchestrator"] = None
    with TestClient(main.app) as c:
        resp = c.post("/chat", json={"message": "Oi", "user_id": "cliente1988"})
    assert resp.status_code == 503 and "OPENAI_API_KEY" in resp.json()["detail"]
