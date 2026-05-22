.PHONY: install shell test lint pipeline features train dvc-push \
        serve api-test compute-baseline drift-report monitor fetch-model \
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
#
# All demo targets delegate to the cross-platform launcher demo/launch.py, so
# they run identically from Windows PowerShell/cmd and macOS/Linux shells.
# Each recipe is a single plain command — no POSIX-shell syntax — so `make`
# works even when it executes recipes through Windows cmd.exe.
# Override PYTHON if `python` is not on PATH (e.g. `make demo PYTHON=python3`).
# ─────────────────────────────────────────────────────────────────────────────

PYTHON ?= python

demo: ## Fetch the model, build/pull images, and start the full demo stack
	$(PYTHON) demo/launch.py up

demo-down: ## Stop the demo stack
	$(PYTHON) demo/launch.py down

demo-reset: ## Hard reset — remove containers + volumes + images, then rebuild
	$(PYTHON) demo/launch.py reset

demo-logs: ## Follow logs from all demo services
	$(PYTHON) demo/launch.py logs

fetch-model: ## Download the model checkpoint from the GitHub release if missing
	$(PYTHON) demo/launch.py fetch-model

demo-cloud: ## Deploy demo stack to a remote Docker host (set DOCKER_HOST=ssh://user@host)
	@[ -n "$(DOCKER_HOST)" ] || \
	  (echo "❌  Set DOCKER_HOST=ssh://user@host before running make demo-cloud" && exit 1)
	@echo "Deploying to remote host: $(DOCKER_HOST)"
	docker compose -f demo/docker-compose.demo.yml up --build -d
	@echo "✅ Deployed. Access via your remote host's public IP on ports 8501 / 3000 / 3001 / 9090"
