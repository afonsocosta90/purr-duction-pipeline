.PHONY: install shell test lint pipeline features train dvc-push \
        serve api-test compute-baseline drift-report monitor \
        demo demo-down demo-reset demo-cloud demo-logs

# ─────────────────────────────────────────────────────────────────────────────
# Core development commands
# ─────────────────────────────────────────────────────────────────────────────

install: ## Install dependencies
	poetry install

shell: ## Enter Poetry shell (Poetry 2.0+)
	@echo "=== Poetry 2.0 shell ==="
	@echo "Run this command manually:"
	@echo "   poetry self add poetry-plugin-shell && poetry shell"
	@echo "Or just use 'poetry run <command>' for everything"

test: ## Run test suite
	poetry run pytest

lint: ## Run linters
	poetry run ruff check .
	poetry run black --check .

pipeline: ## Run full DVC pipeline
	PREFECT_UI_ENABLED=false poetry run dvc repro --force

features: ## Run features stage only
	poetry run dvc repro -s features --force

train: ## Run training stage only
	poetry run dvc repro -s train --force

dvc-push: ## Push DVC artifacts to remote
	poetry run dvc push

serve: ## Start production FastAPI server locally (port 3000)
	poetry run python -m src.catops.serving.service

api-test: ## Quick test of the API (uses a sample cat image)
	curl -X POST "http://127.0.0.1:3000/predict" -F "file=@data/processed/cat/Abyssinian_1.jpg" | jq

compute-baseline: ## Compute pixel stats for all training images → monitoring/baseline_stats.csv
	poetry run python -m src.catops.monitoring.compute_baseline

drift-report: ## Run Evidently drift check against the inference log
	poetry run python -m src.catops.monitoring.drift

monitor: ## Start API + Prometheus + Grafana via docker-compose (Phase 8 stack)
	docker compose up --build

# ─────────────────────────────────────────────────────────────────────────────
# Phase 9 — Interactive Demo Stack
# ─────────────────────────────────────────────────────────────────────────────

demo: ## Build and start full demo (FastAPI + Streamlit + Prometheus + Grafana)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║          Am I a Cat? — MLOps Portfolio Demo                 ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Step 1/2 — Building API image (catops-api:demo)…"
	@echo "  (Streamlit image reuses this layer — only built once)"
	@echo ""
	docker compose -f demo/docker-compose.demo.yml build api
	@echo ""
	@echo "  Step 2/2 — Building Streamlit image (reuses API layer) and starting services…"
	@echo ""
	docker compose -f demo/docker-compose.demo.yml build streamlit
	docker compose -f demo/docker-compose.demo.yml up -d
	@echo ""
	@echo "  ✅ Services are starting. Will be available at:"
	@echo ""
	@echo "     🔮 Streamlit UI    →  http://localhost:8501"
	@echo "     🚀 FastAPI docs    →  http://localhost:3000/docs"
	@echo "     📊 Grafana         →  http://localhost:3001  (admin / catops)"
	@echo "     🔥 Prometheus      →  http://localhost:9090"
	@echo ""
	@echo "  Run 'make demo-logs' to follow all service logs."
	@echo "  Run 'make demo-down' to stop when done."
	@echo ""

demo-down: ## Stop the demo stack
	docker compose -f demo/docker-compose.demo.yml down

demo-reset: ## Hard reset — remove containers + volumes, then rebuild from scratch
	@echo "⚠️  This removes all Prometheus and Grafana data volumes."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	docker compose -f demo/docker-compose.demo.yml down -v --remove-orphans
	docker rmi catops-api:demo catops-demo:latest 2>/dev/null || true
	docker compose -f demo/docker-compose.demo.yml build api
	docker compose -f demo/docker-compose.demo.yml build streamlit
	docker compose -f demo/docker-compose.demo.yml up -d
	@echo "✅ Demo stack rebuilt and running."

demo-logs: ## Follow logs from all demo services
	docker compose -f demo/docker-compose.demo.yml logs -f

demo-cloud: ## Deploy demo stack to a remote Docker host (set DOCKER_HOST=ssh://user@host)
	@[ -n "$(DOCKER_HOST)" ] || \
	  (echo "❌  Set DOCKER_HOST=ssh://user@host before running make demo-cloud" && exit 1)
	@echo "Deploying to remote host: $(DOCKER_HOST)"
	docker compose -f demo/docker-compose.demo.yml up --build -d
	@echo "✅ Deployed. Access via your remote host's public IP on ports 8501 / 3000 / 3001 / 9090"
