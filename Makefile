.PHONY: install run ingest test test-live lint eval eval-fake docker-up docker-down docker-test

install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload --port 8000

ingest:
	python -m app.rag.store

test:
	pytest

test-live:
	RUN_LIVE=1 pytest -m live

lint:
	ruff check app tests evals

eval:
	python -m evals.run_evals

eval-fake:
	python -m evals.run_evals --fake

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-test:
	docker compose run --rm -e OPENAI_API_KEY= api pytest
