.PHONY: pipeline features train

pipeline: ## Run full pipeline (ingest → validate → features → train)
	PREFECT_UI_ENABLED=false poetry run dvc repro --force

features: ## Run only features stage
	poetry run dvc repro -s features --force

train: ## Run only the training stage
	poetry run dvc repro -s train --force