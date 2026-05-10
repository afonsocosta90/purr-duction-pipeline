"""Compute pixel statistics for all training images and save to monitoring/baseline_stats.csv.

Run once after training (or after adding new data):
    poetry run python -m src.catops.monitoring.compute_baseline
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = Path("monitoring/baseline_stats.csv")

COLUMNS = [
    "image_hash",
    "pixel_mean_r",
    "pixel_mean_g",
    "pixel_mean_b",
    "pixel_std_r",
    "pixel_std_g",
    "pixel_std_b",
    "predicted_label",
]


def compute_stats(image_path: Path) -> dict | None:
    try:
        image_bytes = image_path.read_bytes()
        image = Image.open(image_path).convert("RGB")
        arr = np.array(image, dtype=np.float32)
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        return {
            "image_hash": hashlib.md5(image_bytes).hexdigest(),
            "pixel_mean_r": round(float(mean[0]), 4),
            "pixel_mean_g": round(float(mean[1]), 4),
            "pixel_mean_b": round(float(mean[2]), 4),
            "pixel_std_r": round(float(std[0]), 4),
            "pixel_std_g": round(float(std[1]), 4),
            "pixel_std_b": round(float(std[2]), 4),
        }
    except UnidentifiedImageError:
        return None
    except Exception as e:
        print(f"  ⚠ skipping {image_path.name}: {e}", file=sys.stderr)
        return None


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    image_paths: list[tuple[Path, str]] = []
    for label in ("cat", "not_cat"):
        class_dir = PROCESSED_DIR / label
        if not class_dir.exists():
            print(f"⚠ {class_dir} not found — run the pipeline first")
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in class_dir.glob(ext):
                image_paths.append((p, label))

    if not image_paths:
        print("❌ No images found. Run 'make pipeline' first.")
        sys.exit(1)

    print(f"Computing baseline stats for {len(image_paths)} training images...")

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        skipped = 0
        for image_path, label in tqdm(image_paths, desc="Processing"):
            stats = compute_stats(image_path)
            if stats is None:
                skipped += 1
                continue
            stats["predicted_label"] = label
            writer.writerow(stats)

    total = len(image_paths) - skipped
    print(f"✅ Baseline saved to {OUTPUT_PATH} ({total} images, {skipped} skipped)")


if __name__ == "__main__":
    main()
