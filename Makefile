.PHONY: help dev dev-frontend dev-backend test test-frontend test-backend lint lint-frontend lint-backend typecheck migrate docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make dev            Run frontend and backend dev servers"
	@echo "  make dev-frontend   Run Next.js dev server"
	@echo "  make dev-backend    Run FastAPI uvicorn dev server"
	@echo "  make test           Run full Turing_Test_Suite (backend + frontend)"
	@echo "  make test-frontend  Run frontend tests (Turing_Test_Suite/frontend)"
	@echo "  make test-backend   Run backend tests (Turing_Test_Suite/backend)"
	@echo "  make lint           Run linters (eslint, ruff)"
	@echo "  make typecheck      Run type checkers (tsc, mypy)"
	@echo "  make migrate        Run database migrations"
	@echo "  make docker-up      Start all services with docker-compose"
	@echo "  make docker-down    Stop all docker-compose services"
	@echo "  make clean          Clean build artifacts and caches"

dev-frontend:
	npm run dev

dev-backend:
	cd python-engine && uvicorn main:app --reload --port 8000

test: test-backend test-frontend

test-backend:
	python -m pytest Turing_Test_Suite/backend/

test-frontend:
	cd Turing_Test_Suite/frontend && npm test

lint: lint-frontend lint-backend

lint-frontend:
	npm run lint

lint-backend:
	ruff check python-engine/

typecheck:
	npm run typecheck
	mypy python-engine/

migrate:
	cd python-engine && python -c "from database.database import init_db; import asyncio; asyncio.run(init_db())"

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf .next node_modules .pytest_cache python-engine/__pycache__
