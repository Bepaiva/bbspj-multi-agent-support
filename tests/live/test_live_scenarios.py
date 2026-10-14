"""Live end-to-end checks against the real OpenAI model and the persisted knowledge base.

Skipped by default. Run with:  OPENAI_API_KEY=sk-... pytest -m live
"""

import os

import pytest

from tests.unit.test_router_and_rag import SCENARIOS

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not (os.getenv("RUN_LIVE") and os.getenv("LIVE_OPENAI_API_KEY")),
                       reason="set RUN_LIVE=1 and OPENAI_API_KEY to run live tests"),
]


@pytest.fixture(scope="module")
def live_orchestrator(tmp_path_factory):
    from app.core.config import Settings
    from app.graph.orchestrator import SupportOrchestrator
    from app.llm.factory import get_chat_model
    from app.rag.store import KnowledgeStore
    from app.services.customer_repository import get_customer_repository
    from app.services.handoff import TicketStore

    settings = Settings(embeddings_provider="openai", openai_api_key=os.environ["LIVE_OPENAI_API_KEY"],
                        chroma_dir=tmp_path_factory.mktemp("chroma"))
    store = KnowledgeStore(settings=settings)
    store.ingest(scrape_web=False)
    return SupportOrchestrator(llm=get_chat_model(settings=settings), store=store, repo=get_customer_repository(),
                               tickets=TicketStore(tmp_path_factory.mktemp("h") / "h.jsonl"), settings=settings)


@pytest.mark.parametrize("message,expected_route", SCENARIOS)
def test_example_scenarios_live(live_orchestrator, message, expected_route):
    r = live_orchestrator.invoke(message, "cliente1988", session_id=f"live-{hash(message)}")
    assert r["route"] == expected_route
    assert r["response"] and not r["guardrails"]["blocked"]
