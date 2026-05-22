"""Shared MLflow configuration for the training and evaluation stages.

Resolves the tracking URI consistently:

- With DagsHub credentials present (CI), tracking goes to the remote DagsHub
  MLflow server — unchanged behaviour.
- Otherwise it falls back to a local **sqlite** database at the repo root.
  sqlite, unlike the MLflow file store, supports the model registry, so the
  serving layer can resolve ``models:/am-i-a-cat@staging`` offline.

For the training stage (``pin_artifacts=True``) the experiment's artifact
location is pinned to ``<repo>/mlartifacts`` so the API container — which
bind-mounts the same project root — resolves registered model files with no
network access.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

# This file is src/catops/common/mlflow_setup.py → the repo root is 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_DAGSHUB_URI = "https://dagshub.com/afonsocosta90/purr-duction-pipeline.mlflow"
_EXPERIMENT = "am-i-a-cat"


def _sqlite_uri() -> str:
    """Absolute sqlite URI for the local registry DB at the repo root."""
    return f"sqlite:///{(_REPO_ROOT / 'mlflow.db').as_posix()}"


def _ensure_experiment(pin_artifacts: bool) -> None:
    """Get-or-create the ``am-i-a-cat`` experiment on the local sqlite backend.

    NOTE: an experiment's ``artifact_location`` is immutable once created. If a
    stale ``mlflow.db`` exists with a different location, delete ``mlflow.db``
    and ``mlartifacts/`` (both git/dvc-ignored) and re-run.
    """
    client = MlflowClient()
    if client.get_experiment_by_name(_EXPERIMENT) is None:
        artifact_location = (
            (_REPO_ROOT / "mlartifacts").as_uri() if pin_artifacts else None
        )
        client.create_experiment(_EXPERIMENT, artifact_location=artifact_location)
    mlflow.set_experiment(_EXPERIMENT)


def configure_mlflow(pin_artifacts: bool) -> str:
    """Set the MLflow tracking URI + experiment; return the resolved URI.

    ``pin_artifacts`` should be True for the training stage so the registry's
    model artifacts land on the bind-mounted project root.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", _DAGSHUB_URI)
    has_credentials = bool(
        os.environ.get("MLFLOW_TRACKING_USERNAME")
        and os.environ.get("MLFLOW_TRACKING_PASSWORD")
    )
    if not has_credentials and tracking_uri.startswith("http"):
        tracking_uri = _sqlite_uri()

    mlflow.set_tracking_uri(tracking_uri)

    if tracking_uri.startswith("sqlite"):
        _ensure_experiment(pin_artifacts)
    else:
        # DagsHub remote path — keep the original best-effort behaviour.
        try:
            mlflow.set_experiment(_EXPERIMENT)
        except Exception as exc:  # network dependent
            print(
                f"⚠️  Remote MLflow unavailable ({exc.__class__.__name__}), "
                "falling back to local sqlite tracking"
            )
            tracking_uri = _sqlite_uri()
            mlflow.set_tracking_uri(tracking_uri)
            _ensure_experiment(pin_artifacts)

    return tracking_uri
