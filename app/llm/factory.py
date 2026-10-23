"""LLM and embedding factories.

Every agent obtains its model through these functions, which keeps the provider swappable
(and lets tests inject deterministic fakes via ``app.graph.builder.build_graph``).
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings, get_settings


class MissingAPIKeyError(RuntimeError):
    pass


def get_chat_model(model: str | None = None, settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        # Offline development only: a rule-based model that calls the real tools and echoes
        # their results. Lets the whole graph run without network access to an LLM provider.
        from tests.fakes import PolicyChatModel

        return PolicyChatModel(calls=[])
    if not settings.openai_api_key:
        raise MissingAPIKeyError("OPENAI_API_KEY is not set - see .env.example (or set LLM_PROVIDER=fake offline)")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or settings.chat_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


class HashEmbeddings(Embeddings):
    """Deterministic, dependency-free bag-of-words embeddings.

    Not semantically strong, but good enough for offline tests/CI where calling a paid
    embedding API is undesirable. Production uses OpenAI embeddings.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    if settings.embeddings_provider == "hash":
        return HashEmbeddings()
    if not settings.openai_api_key:
        raise MissingAPIKeyError("OPENAI_API_KEY is not set - see .env.example")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
