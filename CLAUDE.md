# CLAUDE.md

Multi-agent customer support system for Getnet merchants (AI Hardcore Engineer challenge).
LangGraph orchestration + FastAPI. Full design doc: `README.md`.

## Setup
- `./scripts/setup_dev.sh` - idempotent: `.venv`, dependencies, `.env`. Run it at the start of a
  session if `.venv` is missing (or wire it as the cloud environment's setup script / a SessionStart hook).
- No `OPENAI_API_KEY` -> `.env` is created in **offline mode** (`LLM_PROVIDER=fake`,
  `EMBEDDINGS_PROVIDER=hash`, web search off). The real graph, tools, RAG and guardrails run;
  only the model is the scripted `tests/fakes.py::PolicyChatModel`.
- Real model: provide `OPENAI_API_KEY` and remove the offline block from `.env`. The cloud
  environment must allow `api.openai.com` (and optionally `site.getnet.com.br`, `www.getnet.net`,
  `duckduckgo.com` / `api.tavily.com`) - the default allowlist blocks them.

## Commands
- Tests (offline, ~2 s): `.venv/bin/pytest`  (live: `RUN_LIVE=1 .venv/bin/pytest -m live`)
- Lint: `.venv/bin/ruff check app tests evals`
- Evals: `.venv/bin/python -m evals.run_evals --fake` (offline) or without `--fake` (real LLM + judge)
- API: `.venv/bin/uvicorn app.main:app --reload --port 8000`, then
  `curl -s localhost:8000/chat -H 'content-type: application/json' -d '{"message":"Oi","user_id":"cliente1988"}'`
- Rebuild the index: `.venv/bin/python -m app.rag.store --no-web`
- Docker is not available in cloud sessions (no daemon); CI builds the image.

## Layout
- `app/graph/orchestrator.py` - LangGraph nodes/edges: input_guardrail -> router -> knowledge|support|escalation -> finalize
- `app/agents/` - `base.py` tool-calling loop, `router.py` (structured output + keyword fallback), `specialists.py`
- `app/prompts/` - all system prompts. `app/tools/` - agent tools. `app/guardrails/checks.py`
- `app/rag/` - sources, scraper, chunking, Chroma store. `data/knowledge_base/*.md` - curated corpus
- `app/services/` - customer repository (mock `data/customers.json`), business days, tickets/handoff
- `tests/unit`, `tests/integration` (real graph + fake LLM), `tests/live` (opt-in), `evals/`

## Conventions
- Every behaviour change ships with a test; keep `pytest` green and offline (no network in tests).
- Support tools must stay scoped server-side to `ctx.user_id` - never add a user-id parameter to a tool.
- New routes/agents: add an eval case to `evals/dataset.jsonl`.
- Prompts are English; agents answer in the user's language (PT/EN).
- Ruff, line length 120. Commit messages in English, imperative.
