# pipelines/data_pipeline.py
from prefect import flow
from src.catops.data.tasks import run_dvc_stage, validate_metadata_exists


@flow(name="am-i-a-cat-data-pipeline", log_prints=True)
def data_pipeline() -> None:
    """End-to-end data orchestration flow — 100% local & free."""
    # Run DVC stages (caching still works perfectly)
    run_dvc_stage("ingest")
    run_dvc_stage("validate")

    # Final quality gate
    metadata = validate_metadata_exists()

    print(f"✅ Pipeline completed successfully! Metadata: {metadata}")
    # Future phases will extend this flow with training, etc.


if __name__ == "__main__":
    data_pipeline()
