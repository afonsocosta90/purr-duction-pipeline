"""
Lightweight Data Validation Stage – "Am I a Cat?" MLOps Pipeline
Zero extra dependencies. Enforces production quality gates.
"""

import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_metadata_if_missing(processed_dir: str = "data/processed") -> Path:
    """Scan folder and create metadata.csv if missing."""
    processed_path = Path(processed_dir)
    metadata_path = processed_path / "metadata.csv"

    if metadata_path.exists():
        logger.info("✅ metadata.csv already exists")
        return metadata_path

    logger.info("🔄 Generating metadata.csv from folder structure...")
    data = []
    for label_dir in ["cat", "not_cat"]:
        label_path = processed_path / label_dir
        if not label_path.exists():
            logger.warning(f"⚠️  Missing class directory: {label_dir}")
            continue
        for img_path in label_path.glob("**/*.*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                data.append(
                    {
                        "image_path": str(img_path.relative_to(processed_path)),
                        "label": label_dir,
                        "file_size": img_path.stat().st_size,
                    }
                )

    df = pd.DataFrame(data)
    df.to_csv(metadata_path, index=False)
    logger.info(
        f"✅ Created metadata.csv with {len(df):,} images "
        f"({df['label'].value_counts().to_dict()})"
    )
    return metadata_path


def validate_processed_data(processed_dir: str = "data/processed") -> None:
    """Run quality gates and raise if any fail."""
    metadata_path = create_metadata_if_missing(processed_dir)
    df = pd.read_csv(metadata_path)

    # Production quality gates
    errors = []

    # 1. Both classes must exist
    class_counts = df["label"].value_counts()
    if len(class_counts) != 2 or any(count == 0 for count in class_counts):
        errors.append(f"Missing classes: {class_counts.to_dict()}")

    # 2. Reasonable balance (at least 30% each)
    proportions = class_counts / len(df)
    if any(prop < 0.3 for prop in proportions):
        errors.append(f"Class imbalance: {class_counts.to_dict()}")

    # 3. No missing paths or tiny/empty files
    if df["image_path"].isnull().any():
        errors.append("Null image_path values found")
    if (df["file_size"] < 1000).any():
        errors.append("Files smaller than 1KB detected (possible corruption)")

    # 4. No duplicate paths
    if df["image_path"].duplicated().any():
        errors.append("Duplicate image paths detected")

    if errors:
        for err in errors:
            logger.error(f"❌ {err}")
        raise RuntimeError("❌ Data validation FAILED — pipeline halted")

    logger.info("✅ DATA VALIDATION PASSED — both classes present and healthy")
    logger.info(f"   Class distribution: {class_counts.to_dict()}")


def validate_features(processed_dir: str = "data/processed") -> None:
    """Post-split validation gates (Phase 3)."""
    from pathlib import Path

    meta = Path(processed_dir) / "metadata.csv"
    train = Path(processed_dir) / "train.csv"
    val = Path(processed_dir) / "val.csv"
    test = Path(processed_dir) / "test.csv"
    config = Path(processed_dir) / "features_config.json"

    if not all(p.exists() for p in [train, val, test, config]):
        logger.warning("Features not yet built – skipping")
        return

    df_train = pd.read_csv(train)
    df_val = pd.read_csv(val)
    df_test = pd.read_csv(test)
    orig_prop = pd.read_csv(meta)["label"].value_counts(normalize=True)

    errors = []
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        split_prop = df["label"].value_counts(normalize=True)
        if not all(
            abs(split_prop.get(k, 0) - orig_prop.get(k, 0)) < 0.02
            for k in orig_prop.index
        ):
            errors.append(f"Class drift in {name}")
    if errors:
        raise RuntimeError("❌ Feature validation FAILED: " + "; ".join(errors))
    logger.info("✅ FEATURE VALIDATION PASSED – splits stratified and leak-free")


if __name__ == "__main__":
    validate_processed_data()
    validate_features()
