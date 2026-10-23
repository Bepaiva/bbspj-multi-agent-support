from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

# Configure the app for offline tests BEFORE importing it (settings are cached).
_TMP = Path(tempfile.mkdtemp(prefix="getnet-tests-"))
# Keep the developer's real key for opt-in live tests, but never use it in offline tests.
if os.getenv("OPENAI_API_KEY") and not os.getenv("LIVE_OPENAI_API_KEY"):
    os.environ["LIVE_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
os.environ.update({
    "ENVIRONMENT": "test",
    "LLM_PROVIDER": "openai",  # tests inject fakes explicitly; keep the API's 503-without-key path testable
    "EMBEDDINGS_PROVIDER": "hash",
    "CHROMA_DIR": str(_TMP / "chroma"),
    "HANDOFF_FILE": str(_TMP / "handoffs.jsonl"),
    "INGEST_ON_STARTUP": "false",
    "INGEST_SCRAPE_WEB": "false",
    "OPENAI_API_KEY": "",
    "LOG_JSON": "false",
    "LOG_LEVEL": "WARNING",
    "RAG_MIN_RELEVANCE": "0.1",
})

import pytest  # noqa: E402

from app.core.config import PROJECT_ROOT, Settings  # noqa: E402
from app.graph.orchestrator import SupportOrchestrator  # noqa: E402
from app.llm.factory import HashEmbeddings  # noqa: E402
from app.rag.store import KnowledgeStore  # noqa: E402
from app.services.customer_repository import CustomerRepository  # noqa: E402
from app.services.handoff import TicketStore  # noqa: E402
from tests.fakes import PolicyChatModel  # noqa: E402

# Wednesday 14/10/2026 -> "yesterday" is Tuesday 13/10/2026
FIXED_TODAY = date(2026, 10, 14)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def kb_store(settings: Settings) -> KnowledgeStore:
    s = settings.model_copy(update={"chroma_collection": "test_kb"})
    store = KnowledgeStore(settings=s, embeddings=HashEmbeddings(), persist=False)
    store.ingest(scrape_web=False)
    return store


@pytest.fixture
def repo() -> CustomerRepository:
    return CustomerRepository(PROJECT_ROOT / "data" / "customers.json", PROJECT_ROOT / "data" / "decline_codes.json",
                              today=FIXED_TODAY)


@pytest.fixture
def tickets(tmp_path: Path) -> TicketStore:
    return TicketStore(tmp_path / "handoffs.jsonl")


@pytest.fixture
def fake_llm() -> PolicyChatModel:
    return PolicyChatModel(calls=[])


@pytest.fixture(autouse=True)
def fake_web_search(monkeypatch):
    def _search(query, max_results=None, settings=None):
        return [{"title": f"Result for {query}", "url": "https://example.com/result", "snippet": "EUR/BRL 6.21 today"}]

    monkeypatch.setattr("app.tools.knowledge_tools.web_search", _search)


@pytest.fixture
def orchestrator(fake_llm, kb_store, repo, tickets, settings) -> SupportOrchestrator:
    return SupportOrchestrator(llm=fake_llm, store=kb_store, repo=repo, tickets=tickets, settings=settings)
