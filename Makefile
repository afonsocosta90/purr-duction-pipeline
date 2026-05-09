.PHONY: install shell test lint pipeline features train dvc-push

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
