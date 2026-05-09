"""Production-grade evaluation module for "Am I a Cat?" pipeline.
- Loads model + val/test CSVs via CatDataset
- Computes full classification metrics using sklearn
- Generates and logs confusion matrix + ROC curve as MLflow artifacts
- Returns metrics dict for promotion logic + tracking
"""

from pathlib import Path
import json
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import torch
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from src.catops.data.dataset import CatDataset


def evaluate_model(
    model: torch.nn.Module,
    cfg: DictConfig,
    processed_dir: Path = Path("data/processed"),
    split: str = "val",
) -> Dict[str, float]:
    """Evaluate model on specified split and log artifacts to MLflow."""
    features_config = json.loads((processed_dir / "features_config.json").read_text())

    # === FIXED: Correct torchvision transforms ===
    transform = transforms.Compose(
        [
            transforms.Resize(tuple(features_config["resize"]["target_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=features_config["normalization"]["mean"],
                std=features_config["normalization"]["std"],
            ),
        ]
    )

    # Load dataset for the split
    csv_path = processed_dir / f"{split}.csv"
    dataset = CatDataset(
        csv_path=csv_path,
        transform=transform,  # Pass transform directly (more efficient)
        processed_dir=processed_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.backends.mps.is_available() or torch.cuda.is_available(),
    )

    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # prob of "not_cat" (class 1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    y_true = all_labels
    y_pred = all_preds
    y_prob = all_probs

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "precision": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

    # Log to MLflow
    mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics.items()})

    # Confusion Matrix artifact
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["cat", "not_cat"],
        yticklabels=["cat", "not_cat"],
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix - {split.upper()} Split")
    cm_path = Path("artifacts") / f"confusion_matrix_{split}.png"
    cm_path.parent.mkdir(exist_ok=True)
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

    # ROC Curve artifact
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {split.upper()} Split")
    plt.legend()
    roc_path = Path("artifacts") / f"roc_curve_{split}.png"
    plt.savefig(roc_path)
    plt.close()
    mlflow.log_artifact(str(roc_path), artifact_path="evaluation")

    print(
        f"✅ {split.upper()} evaluation complete → "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    return metrics


# For standalone testing
if __name__ == "__main__":
    print("✅ Evaluation module ready for import from train.py")
