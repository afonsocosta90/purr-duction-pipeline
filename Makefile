# Phase 3: Pipeline Orchestration
.PHONY: pipeline
pipeline: ## Run full data pipeline locally with Prefect (cleaner execution)
	PREFECT_UI_ENABLED=false poetry run python -m pipelines.data_pipeline

.PHONY: prefect-ui
prefect-ui: ## Start local Prefect dashboard (free, run in separate terminal)
	poetry run prefect server start --port 4200