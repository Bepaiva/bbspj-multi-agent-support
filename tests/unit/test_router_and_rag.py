import pytest

from app.agents.router import RouterAgent, keyword_route
from app.rag.loaders import WebScraper, load_seed_corpus
from app.rag.store import chunk_documents
from tests.fakes import FailingChatModel

# The challenge's example scenarios and the route each one must take.
SCENARIOS = [
    ("What's the difference between the Get Clássica and the Get Smart?", "knowledge"),
    ("What's the weather forecast in Porto Alegre tomorrow?", "knowledge"),
    ("When will the money from yesterday's sales be deposited?", "support"),
    ("Do I need a bank account to receive my sales via Pix?", "knowledge"),
    ("My card machine won't connect to the internet, what should I do?", "support"),
    ("How does receivables advance (antecipação) work with Getnet?", "knowledge"),
    ("What's the euro exchange rate today?", "knowledge"),
    ("My card machine is showing a transaction decline error.", "support"),
    ("How many installments can I split a sale into with the crediário?", "knowledge"),
    ("Can I sell through WhatsApp using the Payment Link?", "knowledge"),
]


@pytest.mark.parametrize("message,expected", SCENARIOS)
def test_keyword_fallback_routes_example_scenarios(message, expected):
    assert keyword_route(message).route == expected


def test_router_falls_back_when_llm_fails():
    decision, fallback = RouterAgent(FailingChatModel()).route([])
    assert fallback is True
    assert decision.route == "knowledge"


def test_seed_corpus_has_sources(settings):
    docs = load_seed_corpus(settings.seed_kb_dir)
    assert len(docs) >= 8
    assert all(d.metadata["source"].startswith("https://") for d in docs)


def test_chunks_keep_title_and_section(settings):
    docs = load_seed_corpus(settings.seed_kb_dir)
    chunks = chunk_documents(docs, 900, 150)
    assert len(chunks) > len(docs)
    assert all(c.page_content.startswith("[") and c.metadata["chunk_id"] for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


@pytest.mark.parametrize("query,expected_title", [
    ("difference between Get Clássica and Get Smart", "card machines"),
    ("Do I need a bank account to receive Pix", "Pix"),
    ("sell through WhatsApp payment link", "Payment Link"),
    ("crediário installments 48x", "Crediário"),
    ("antecipação de recebíveis how it works", "antecipação"),
])
def test_retrieval_finds_the_right_document(kb_store, query, expected_title):
    hits = kb_store.search(query, k=3)
    assert hits, query
    assert expected_title.lower() in hits[0][0].metadata["title"].lower()


def test_ingestion_is_idempotent(kb_store):
    before = kb_store.count()
    kb_store.ingest(scrape_web=False)
    assert kb_store.count() == before


def test_scraper_extracts_main_text_without_navigation():
    html = """<html><head><title>Pix | Getnet</title><script>var x=1</script></head>
    <body><nav>Menu Home Produtos</nav><main><h1>Pix</h1><p>Receba suas vendas em segundos com Pix.</p></main>
    <footer>Copyright</footer></body></html>"""
    title, text = WebScraper.extract_text(html)
    assert title == "Pix | Getnet"
    assert "Receba suas vendas" in text and "Menu" not in text and "Copyright" not in text
