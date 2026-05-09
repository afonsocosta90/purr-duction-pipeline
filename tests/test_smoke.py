"""Smoke tests — verify key modules import and basic contracts hold."""

from pathlib import Path


def test_catdataset_importable():
    from src.catops.data.dataset import CatDataset

    assert CatDataset is not None


def test_evaluate_importable():
    from src.catops.evaluation.evaluate import evaluate_model

    assert evaluate_model is not None


def test_processed_splits_exist():
    """Fail fast if the DVC-produced CSVs are missing (skipped when data not pulled)."""
    import pytest

    processed = Path("data/processed")
    if not processed.exists():
        pytest.skip("data/processed not present — run `dvc repro` first")
    for split in ("train.csv", "val.csv", "test.csv"):
        assert (processed / split).exists(), f"Missing {split} — run `dvc repro`"
