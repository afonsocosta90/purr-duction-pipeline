.PHONY: install shell test lint pipeline features train dvc-push

install: ## Install dependencies
	poetry install

shell: ## Enter Poetry shell
	poetry shell

test: ## Run test suite
	poetry run pytest

lint: ## Run linters & pre-commit
	poetry run pre-commit run --all-files

pipeline: ## Run full DVC pipeline
	PREFECT_UI_ENABLED=false poetry run dvc repro --force

features: ## Run features stage only
	poetry run dvc repro -s features --force

train: ## Run training stage only
	poetry run dvc repro -s train --force

dvc-push: ## Push DVC artifacts to remote
	poetry run dvc push