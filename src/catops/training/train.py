"""Production Model training stage for "Am I a Cat?" pipeline.

- Uses CatDataset (stratified splits)
- Real evaluation on val split via dedicated evaluate.py
- Full MLflow tracking + artifacts + promotion based on REAL metrics
- Modern weights API
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import mlflow
import mlflow.pytorch
import hydra
from omegaconf import DictConfig
import json

from src.catops.data.dataset import CatDataset
from src.catops.evaluation.evaluate import evaluate_model  # NEW


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    processed_dir = Path("data/processed")
    features_config = json.loads((processed_dir / "features_config.json").read_text())

    torch.manual_seed(cfg.training.seed)

    # Transforms
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

    # Train loader
    train_dataset = CatDataset(
        csv_path=processed_dir / "train.csv",
        transform=transform,
        processed_dir=processed_dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # Model (modern API)
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if cfg.model.pretrained else None
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(cfg.model.dropout),
        nn.Linear(num_ftrs, cfg.model.num_classes),
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # MLflow
    mlflow.set_experiment("am-i-a-cat")
    with mlflow.start_run(run_name=f"resnet50-seed-{cfg.training.seed}"):
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
                "lr": cfg.training.learning_rate,
                "train_samples": len(train_dataset),
                "device": str(device),
            }
        )

        print(f"🚀 Training on {device} for {cfg.training.epochs} epochs...")

        for epoch in range(cfg.training.epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{cfg.training.epochs} - Loss: {avg_loss:.4f}")

        # === REAL EVALUATION (Phase 5) ===
        val_metrics = evaluate_model(model, cfg, processed_dir, split="val")

        # Log final metrics
        mlflow.log_metrics(
            {
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )

        # === AUTOMATED PROMOTION BASED ON REAL VAL METRICS ===
        if (
            val_metrics["accuracy"] >= cfg.training.promotion.min_accuracy
            and val_metrics["f1"] >= cfg.training.promotion.min_f1
        ):
            mlflow.pytorch.log_model(model, "model")
            print("🎉 MODEL PROMOTED TO STAGING – REAL thresholds met!")
            mlflow.set_tag("stage", "staging")
        else:
            print("⚠️  Model did NOT meet promotion thresholds.")

        # DVC output
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), models_dir / "best_model.pt")
        (models_dir / "model_config.json").write_text(
            json.dumps(
                {"num_classes": cfg.model.num_classes, "dropout": cfg.model.dropout}
            )
        )
        print("💾 Model saved locally as models/best_model.pt")

        print("✅ Training + Evaluation completed & fully logged to MLflow.")


if __name__ == "__main__":
    train()
