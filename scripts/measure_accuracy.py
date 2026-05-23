"""Measure accuracy / F1 / precision / recall / AUC on val + test splits.

Loads the current `models/best_model.pt` and runs both DVC splits through
it. No MLflow / no artifact writes — just prints measured metrics for the
README "Project KPIs" section.

Run from the repo root:

    poetry run python scripts/measure_accuracy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make repo root importable so `src.catops.*` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import models as tv_models, transforms

from src.catops.data.dataset import CatDataset

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/best_model.pt")
MODEL_CONFIG = Path("models/model_config.json")


def build_model() -> torch.nn.Module:
    cfg = json.loads(MODEL_CONFIG.read_text())
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(cfg["dropout"]),
        nn.Linear(model.fc.in_features, cfg["num_classes"]),
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    return model.to(device).eval()


def build_transform() -> transforms.Compose:
    feat = json.loads((PROCESSED_DIR / "features_config.json").read_text())
    return transforms.Compose(
        [
            transforms.Resize(tuple(feat["resize"]["target_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=feat["normalization"]["mean"],
                std=feat["normalization"]["std"],
            ),
        ]
    )


def evaluate(model: torch.nn.Module, transform, split: str) -> dict:
    ds = CatDataset(
        csv_path=PROCESSED_DIR / f"{split}.csv",
        transform=transform,
        processed_dir=PROCESSED_DIR,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    device = next(model.parameters()).device

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "precision": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def main() -> None:
    model = build_model()
    transform = build_transform()
    print(f"Device: {next(model.parameters()).device}")
    print(
        f"{'split':>5}  {'n':>5}  {'acc':>7}  {'f1':>7}  {'prec':>7}  {'rec':>7}  {'auc':>7}"
    )
    for split in ("val", "test"):
        m = evaluate(model, transform, split)
        print(
            f"{split:>5}  {m['n']:5d}  {m['accuracy']:7.4f}  "
            f"{m['f1']:7.4f}  {m['precision']:7.4f}  "
            f"{m['recall']:7.4f}  {m['roc_auc']:7.4f}"
        )


if __name__ == "__main__":
    main()
