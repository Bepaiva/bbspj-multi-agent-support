"""FastAPI application: POST /chat plus operational endpoints."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.schemas import ChatRequest, ChatResponse
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger, new_trace_id, set_trace_id
from app.core.metrics import REQUEST_LATENCY, REQUESTS
from app.graph.orchestrator import SupportOrchestrator
from app.llm.factory import MissingAPIKeyError, get_chat_model
from app.rag.store import get_knowledge_store
from app.services.customer_repository import get_customer_repository
from app.services.handoff import get_ticket_store

settings = get_settings()
configure_logging(settings.log_level, settings.log_json)
log = get_logger("app")

_state: dict = {"orchestrator": None, "ingest": {"status": "idle"}}
_ingest_lock = threading.Lock()


def build_orchestrator() -> SupportOrchestrator:
    return SupportOrchestrator(
        llm=get_chat_model(settings.chat_model),
        router_llm=get_chat_model(settings.effective_router_model),
        store=get_knowledge_store(),
        repo=get_customer_repository(),
        tickets=get_ticket_store(),
    )


def get_orchestrator() -> SupportOrchestrator:
    if _state["orchestrator"] is None:
        _state["orchestrator"] = build_orchestrator()
    return _state["orchestrator"]


def run_ingestion(scrape_web: bool | None = None) -> dict:
    with _ingest_lock:
        _state["ingest"] = {"status": "running"}
        try:
            stats = get_knowledge_store().ingest(scrape_web=scrape_web)
            _state["ingest"] = {"status": "done", **stats}
        except Exception as exc:
            log.error("ingest.failed", error=str(exc))
            _state["ingest"] = {"status": "failed", "error": str(exc)}
        return _state["ingest"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        store = get_knowledge_store()
        count = store.count()
        log.info("startup.knowledge_base", chunks=count)
        if count == 0 and settings.ingest_on_startup:
            # Seed corpus first (fast) so the API is useful immediately, then scrape the web in background.
            run_ingestion(scrape_web=False)
            if settings.ingest_scrape_web:
                threading.Thread(target=run_ingestion, kwargs={"scrape_web": True}, daemon=True).start()
    except MissingAPIKeyError as exc:
        log.error("startup.missing_api_key", error=str(exc))
    yield


app = FastAPI(
    title="Getnet Multi-Agent Support",
    version="1.0.0",
    description="Router + Knowledge (RAG/web) + Customer Support + Escalation agents orchestrated with LangGraph.",
    lifespan=lifespan,
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    structlog.contextvars.clear_contextvars()
    incoming = request.headers.get("x-trace-id")
    trace_id = set_trace_id(incoming[:64]) if incoming else new_trace_id()
    structlog.contextvars.bind_contextvars(trace_id=trace_id, path=request.url.path)
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-trace-id"] = trace_id
    log.info("http.request", method=request.method, status=response.status_code,
             latency_ms=int((time.perf_counter() - start) * 1000))
    return response


@app.post("/chat", response_model=ChatResponse, summary="Send a message to the multi-agent system")
async def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    try:
        orchestrator = get_orchestrator()
        result = await run_in_threadpool(orchestrator.invoke, req.message, req.user_id, req.session_id)
    except MissingAPIKeyError as exc:
        REQUESTS.labels("none", "error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        REQUESTS.labels("none", "error").inc()
        log.exception("chat.failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Internal error - see logs with the x-trace-id") from exc
    REQUESTS.labels(result["route"] or "none", "ok").inc()
    REQUEST_LATENCY.labels(result["route"] or "none").observe(time.perf_counter() - start)
    log.info("chat.completed", user_id=req.user_id, route=result["route"], agent=result["agent"],
             escalated=result["escalated"], confidence=result["confidence"],
             tools=[t["tool"] for t in result["tool_calls"]], total_ms=result["timings_ms"]["total"])
    return ChatResponse(**result)


@app.post("/admin/ingest", summary="(Re)build the knowledge base")
async def ingest(scrape_web: bool = True, x_admin_token: str | None = Header(default=None)):
    if settings.admin_token and x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="invalid admin token")
    return await run_in_threadpool(run_ingestion, scrape_web)


@app.get("/health", summary="Liveness")
async def health():
    return {"status": "ok"}


@app.get("/ready", summary="Readiness: knowledge base and LLM configuration")
async def ready():
    try:
        chunks = get_knowledge_store().count()
    except Exception as exc:
        return JSONResponse(status_code=503, content={"ready": False, "error": str(exc)})
    ok = chunks > 0 and settings.llm_configured
    return JSONResponse(status_code=200 if ok else 503, content={
        "ready": ok, "knowledge_base_chunks": chunks, "llm_configured": settings.llm_configured,
        "llm_provider": settings.llm_provider, "embeddings_provider": settings.embeddings_provider,
        "ingestion": _state["ingest"], "model": settings.chat_model})


@app.get("/metrics", response_class=PlainTextResponse, summary="Prometheus metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
