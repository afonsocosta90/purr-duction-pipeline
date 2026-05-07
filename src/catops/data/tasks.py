from prefect import task, get_run_logger
from pathlib import Path
import subprocess

@task(retries=3, retry_delay_seconds=60)
def run_dvc_stage(stage_name: str) -> None:
    """Run a DVC stage with retries and logging."""
    logger = get_run_logger()
    logger.info(f"🚀 Running DVC stage: {stage_name}")
    
    result = subprocess.run(
        ["dvc", "repro", "-s", stage_name],
        capture_output=True,
        text=True,
        check=True
    )
    
    logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)

@task
def validate_metadata_exists() -> Path:
    """Simple quality gate after validation stage."""
    meta = Path("data/processed/metadata.csv")
    if not meta.exists():
        raise FileNotFoundError("metadata.csv not found after validation stage")
    return meta