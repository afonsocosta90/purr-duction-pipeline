.PHONY: pipeline features
pipeline: ## Run full pipeline (ingest → validate → features)
	PREFECT_UI_ENABLED=false poetry run dvc repro --force

features: ## Run only features stage
	poetry run dvc repro -s features --force