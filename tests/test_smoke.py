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
    """When the pipeline has run, all three splits must be present.

    Skips when none are present — a fresh clone, or a checkout where the
    demo launcher created data/processed/ to hold features_config.json
    only. The CI quality job never runs `dvc repro`, so it skips there.
    """
    processed = Path("data/processed")
    splits = ("train.csv", "val.csv", "test.csv")
    if not any((processed / split).exists() for split in splits):
        import pytest

        pytest.skip("pipeline splits not present — run `dvc repro` first")
    for split in splits:
        assert (processed / split).exists(), f"Missing {split}"
