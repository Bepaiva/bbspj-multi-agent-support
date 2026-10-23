"""Centralised, typed configuration loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Service -------------------------------------------------------------
    app_name: str = "getnet-multi-agent-support"
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_json: bool = True
    admin_token: str | None = None  # protects POST /admin/ingest when set

    # --- LLM -----------------------------------------------------------------
    # "fake" = deterministic scripted model (tests/fakes.py) for offline development only,
    # e.g. in a cloud workspace without network access to the OpenAI API.
    llm_provider: Literal["openai", "fake"] = "openai"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    chat_model: str = "gpt-4o-mini"
    router_model: str | None = None  # defaults to chat_model; a cheaper model can be used
    llm_temperature: float = 0.1
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    # --- Embeddings / vector store --------------------------------------------
    embeddings_provider: Literal["openai", "hash"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    chroma_dir: Path = PROJECT_ROOT / "storage" / "chroma"
    chroma_collection: str = "getnet_kb"
    seed_kb_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"
    rag_top_k: int = 5
    rag_min_relevance: float = 0.30
    chunk_size: int = 900
    chunk_overlap: int = 150
    ingest_on_startup: bool = True
    ingest_scrape_web: bool = True  # scrape live Getnet pages in addition to the seed corpus

    # --- Web search -----------------------------------------------------------
    web_search_provider: Literal["tavily", "duckduckgo", "none"] = "duckduckgo"
    tavily_api_key: str | None = None
    web_search_max_results: int = 5

    # --- Customer data / handoff -------------------------------------------------
    customers_file: Path = PROJECT_ROOT / "data" / "customers.json"
    decline_codes_file: Path = PROJECT_ROOT / "data" / "decline_codes.json"
    handoff_file: Path = PROJECT_ROOT / "storage" / "handoffs.jsonl"
    handoff_webhook_url: str | None = None  # e.g. a Microsoft Teams / Slack incoming webhook

    # --- Orchestration ---------------------------------------------------------
    max_agent_iterations: int = Field(default=5, ge=1, le=10)
    router_confidence_threshold: float = 0.55
    history_turns: int = 6

    @property
    def llm_configured(self) -> bool:
        return self.llm_provider == "fake" or bool(self.openai_api_key)

    @property
    def effective_router_model(self) -> str:
        return self.router_model or self.chat_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
