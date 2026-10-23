"""Vector store (Chroma, persisted on disk), chunking and ingestion pipeline.

Pipeline: load (seed markdown + scraped pages) -> clean -> chunk (markdown-header aware,
then recursive character split) -> embed -> upsert into Chroma with deterministic ids
(so re-ingesting is idempotent).
"""

from __future__ import annotations

import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.llm.factory import get_embeddings
from app.rag.loaders import WebScraper, load_seed_corpus
from app.rag.sources import GETNET_URLS

log = get_logger(__name__)

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=False)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks: list[Document] = []
    for doc in docs:
        sections = md_splitter.split_text(doc.page_content) if doc.metadata.get("origin") == "seed" else [doc]
        for section in sections:
            heading = " > ".join(section.metadata[k] for k in ("h1", "h2", "h3") if k in section.metadata)
            for piece in char_splitter.split_text(section.page_content):
                meta = {k: v for k, v in doc.metadata.items()}
                meta["section"] = heading
                # Prefix the doc title so every chunk carries its context (helps retrieval).
                content = f"[{doc.metadata['title']}{' - ' + heading if heading else ''}]\n{piece}"
                meta["chunk_id"] = hashlib.sha1(f"{doc.metadata['doc_id']}|{content}".encode()).hexdigest()
                chunks.append(Document(page_content=content, metadata=meta))
    return chunks


class KnowledgeStore:
    def __init__(self, settings: Settings | None = None, embeddings: Embeddings | None = None,
                 persist: bool = True) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embeddings or get_embeddings(self.settings)
        # One collection per embedding provider: vectors from different models have different
        # dimensions, so switching EMBEDDINGS_PROVIDER must not reuse an incompatible index.
        collection = f"{self.settings.chroma_collection}_{self.settings.embeddings_provider}"
        kwargs = {"collection_name": collection, "embedding_function": self.embeddings,
                  "collection_metadata": {"hnsw:space": "cosine"}}
        if persist:
            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            kwargs["persist_directory"] = str(self.settings.chroma_dir)
        self.vectorstore = Chroma(**kwargs)

    def count(self) -> int:
        return self.vectorstore._collection.count()

    def ingest(self, *, scrape_web: bool | None = None, urls: list[str] | None = None) -> dict:
        s = self.settings
        docs = load_seed_corpus(s.seed_kb_dir)
        scrape = s.ingest_scrape_web if scrape_web is None else scrape_web
        web_docs: list[Document] = []
        if scrape:
            web_docs = WebScraper().fetch_many(urls or GETNET_URLS)
        chunks = chunk_documents(docs + web_docs, s.chunk_size, s.chunk_overlap)
        ids = [c.metadata["chunk_id"] for c in chunks]
        # de-duplicate ids (identical chunks across pages)
        unique = {cid: c for cid, c in zip(ids, chunks, strict=False)}
        if unique:
            self.vectorstore.add_documents(list(unique.values()), ids=list(unique.keys()))
        stats = {"seed_documents": len(docs), "web_documents": len(web_docs), "chunks": len(unique),
                 "collection_size": self.count()}
        log.info("ingest.completed", **stats)
        return stats

    def search(self, query: str, k: int | None = None, min_relevance: float | None = None) -> list[tuple[Document, float]]:
        k = k or self.settings.rag_top_k
        threshold = self.settings.rag_min_relevance if min_relevance is None else min_relevance
        # Fetch extra candidates, then drop near-duplicates and low-relevance chunks.
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k * 2)
        seen, out = set(), []
        for doc, score in sorted(results, key=lambda r: r[1], reverse=True):
            key = doc.page_content[:200]
            if score < threshold or key in seen:
                continue
            seen.add(key)
            out.append((doc, score))
            if len(out) >= k:
                break
        return out


_store: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store


def main() -> None:  # python -m app.rag.store
    import argparse

    from app.core.logging import configure_logging

    parser = argparse.ArgumentParser(description="Ingest the Getnet knowledge base into Chroma")
    parser.add_argument("--no-web", action="store_true", help="only ingest the curated seed corpus")
    args = parser.parse_args()
    configure_logging(get_settings().log_level, json_logs=False)
    stats = get_knowledge_store().ingest(scrape_web=not args.no_web)
    print(stats)


if __name__ == "__main__":
    main()
