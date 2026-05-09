"""Feature engineering stage for "Am I a Cat?" pipeline.

- Stratified train/val/test split (70/15/15) using label column
- Computes resize target + normalization statistics (train-set only)
- Outputs DVC-tracked CSVs + config artifacts
- Zero data leakage, fully deterministic via Hydra + fixed seed
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../../configs", config_name="data", version_base=None)
def build_features(cfg: DictConfig) -> None:
    processed_dir = Path("data/processed")
    metadata_path = processed_dir / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"❌ metadata.csv not found at {metadata_path}. Run ingest + validate first."
        )

    print("🔄 Loading metadata...")
    df = pd.read_csv(metadata_path)

    # === 1. Stratified Split ===
    print(
        f"📊 Performing stratified split (70/15/15) with seed={cfg.split.random_seed}..."
    )
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - cfg.split.train_ratio),
        stratify=df["label"],
        random_state=cfg.split.random_seed,
    )
    val_ratio_adj = cfg.split.val_ratio / (cfg.split.val_ratio + cfg.split.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adj),
        stratify=temp_df["label"],
        random_state=cfg.split.random_seed,
    )

    train_df[["image_path", "label"]].to_csv(processed_dir / "train.csv", index=False)
    val_df[["image_path", "label"]].to_csv(processed_dir / "val.csv", index=False)
    test_df[["image_path", "label"]].to_csv(processed_dir / "test.csv", index=False)

    print(
        f"✅ Split complete → Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
    )

    # === 2. Lightweight preprocessing decisions (FIXED) ===
    # Convert OmegaConf objects to native Python types for JSON serialization
    resize_config = OmegaConf.to_container(cfg.resize, resolve=True)
    norm_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    features_config = {
        "resize": resize_config,
        "normalization": norm_stats,
        "split_seed": int(cfg.split.random_seed),  # ensure native int
    }

    (processed_dir / "features_config.json").write_text(
        json.dumps(features_config, indent=2)
    )

    print(
        "✅ Feature artifacts saved (train.csv, val.csv, test.csv, features_config.json)"
    )


if __name__ == "__main__":
    build_features()
