#!/usr/bin/env bash
# Idempotent development setup - safe to run any time.
#
#   ./scripts/setup_dev.sh
#
# Can be wired as the cloud environment's setup script or a Claude Code SessionStart hook, so a
# fresh cloud workspace is ready to code and test in seconds.
#
# What it does:
#   1. creates .venv (if missing) and installs requirements-dev.txt (only when they changed)
#   2. creates .env from .env.example if missing:
#        - OPENAI_API_KEY in the environment  -> real OpenAI mode
#        - no key                             -> offline mode (fake LLM + hash embeddings)
#   3. exposes .venv on PATH for the rest of the session (CLAUDE_ENV_FILE)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { echo "[setup_dev] $*" >&2; }

# 1. Python environment --------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  log "creating .venv"
  if command -v uv >/dev/null 2>&1; then uv venv -q .venv; else python3 -m venv .venv; fi
fi

REQ_HASH="$(cat requirements.txt requirements-dev.txt | sha256sum | cut -c1-16)"
if [ "$(cat .venv/.requirements-hash 2>/dev/null || true)" != "$REQ_HASH" ]; then
  log "installing dependencies"
  if command -v uv >/dev/null 2>&1; then
    uv pip install -q --python .venv/bin/python -r requirements-dev.txt
  else
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -r requirements-dev.txt
  fi
  echo "$REQ_HASH" > .venv/.requirements-hash
else
  log "dependencies up to date"
fi

# 2. .env ------------------------------------------------------------------------
if [ ! -f .env ]; then
  cp .env.example .env
  if [ -n "${OPENAI_API_KEY:-}" ]; then
    # Key comes from the environment; keep it out of the file.
    sed -i 's/^OPENAI_API_KEY=.*/# OPENAI_API_KEY is provided by the environment/' .env
    log "created .env (OpenAI mode - key from environment)"
  else
    sed -i 's/^OPENAI_API_KEY=.*/# OPENAI_API_KEY=sk-...   # set it to use the real model/' .env
    cat >> .env <<'EOF'

# --- Added by scripts/setup_dev.sh: offline mode (no OPENAI_API_KEY found) ---------
LLM_PROVIDER=fake
EMBEDDINGS_PROVIDER=hash
INGEST_SCRAPE_WEB=false
WEB_SEARCH_PROVIDER=none
LOG_JSON=false
EOF
    log "created .env (offline mode: fake LLM + hash embeddings)"
  fi
fi

# 3. Session PATH ------------------------------------------------------------------
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$ROOT/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
  echo "export VIRTUAL_ENV=\"$ROOT/.venv\"" >> "$CLAUDE_ENV_FILE"
fi

log "ready - run: .venv/bin/pytest   |   .venv/bin/uvicorn app.main:app --reload"
