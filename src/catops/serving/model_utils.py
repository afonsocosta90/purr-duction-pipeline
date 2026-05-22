"""Model loading and preprocessing utilities shared between serving and evaluation."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# Canonical class mapping — matches CatDataset label encoding: cat=0, not_cat=1
CLASS_NAMES: dict[int, str] = {0: "cat", 1: "not_cat"}


def _wait_for_file(path: Path, timeout: int = 10) -> None:
    """Poll for a file's existence to handle slow Docker bind-mount sync."""
    start = time.time()
    while not path.exists():
        if time.time() - start > timeout:
            return  # Timeout reached, let the caller handle the missing file
        time.sleep(0.5)


_DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.pt"))
_DEFAULT_CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", "models/model_config.json"))
_DEFAULT_FEATURES_CONFIG_PATH = Path(
    os.getenv("FEATURES_CONFIG_PATH", "data/processed/features_config.json")
)


def build_inference_transform(
    features_config_path: str | Path = _DEFAULT_FEATURES_CONFIG_PATH,
) -> transforms.Compose:
    """Build the inference transform from features_config.json written by the pipeline.

    Reads resize target and normalization stats dynamically so the serving
    transform stays in sync if the features stage changes them.
    """
    path = Path(features_config_path)
    _wait_for_file(path)
    cfg = json.loads(path.read_text())
    return transforms.Compose(
        [
            transforms.Resize(tuple(cfg["resize"]["target_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg["normalization"]["mean"],
                std=cfg["normalization"]["std"],
            ),
        ]
    )


def load_model(
    model_path: str | Path = _DEFAULT_MODEL_PATH,
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
) -> nn.Module:
    """Load a trained ResNet50 from a state-dict checkpoint.

    Reads num_classes and dropout from model_config.json (written by the training
    stage) so the architecture stays in sync when hyperparameters change.
    Falls back to safe defaults for checkpoints created before config saving was added.
    """
    model_path = Path(model_path)
    config_path = Path(config_path)

    _wait_for_file(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        num_classes: int = cfg["num_classes"]
        dropout: float = cfg["dropout"]
    else:
        num_classes = 2
        dropout = 0.2

    base = models.resnet50(weights=None)
    num_ftrs = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, num_classes),
    )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    base.load_state_dict(state_dict)
    base.eval()
    return base


def load_serving_model() -> tuple[nn.Module, str]:
    """Load the serving model, preferring the MLflow model registry.

    When ``MLFLOW_TRACKING_URI`` is set, load ``models:/am-i-a-cat@staging``
    from the registry — this is the registry-driven deployment path: the model
    promoted to the ``staging`` alias by the training stage is what gets
    served, with no manual file copy.

    On any failure — no tracking URI, no model registered under that alias
    yet, or the backend being unreachable — fall back to the on-disk
    checkpoint via :func:`load_model`.

    Returns a ``(model, source)`` tuple where ``source`` is ``"registry"`` or
    ``"disk"``.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        try:
            import mlflow
            import mlflow.pytorch

            mlflow.set_tracking_uri(tracking_uri)
            model = mlflow.pytorch.load_model("models:/am-i-a-cat@staging")
            model.eval()
            return model, "registry"
        except Exception as exc:
            logger.warning(
                "Registry load failed (%s: %s); falling back to disk checkpoint",
                exc.__class__.__name__,
                exc,
            )
    return load_model(), "disk"
