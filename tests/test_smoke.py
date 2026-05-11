"""Smoke tests — verify project structure and lightweight module contracts."""

from pathlib import Path


def test_source_files_exist():
    root = Path("src/catops")
    expected = [
        "data/ingest.py",
        "data/validate.py",
        "data/dataset.py",
        "features/build_features.py",
        "training/train.py",
        "evaluation/evaluate.py",
    ]
    for rel in expected:
        assert (root / rel).exists(), f"Missing source file: src/catops/{rel}"


def test_configs_exist():
    configs = Path("configs")
    for name in ("data.yaml", "model.yaml", "training.yaml"):
        assert (configs / name).exists(), f"Missing config: configs/{name}"


def test_dvc_pipeline_defined():
    assert Path("dvc.yaml").exists()


def test_processed_splits_exist():
    processed = Path("data/processed")
    if not processed.exists():
        import pytest

        pytest.skip("data/processed not present — run `dvc repro` first")
    for split in ("train.csv", "val.csv", "test.csv"):
        assert (processed / split).exists(), f"Missing {split}"
