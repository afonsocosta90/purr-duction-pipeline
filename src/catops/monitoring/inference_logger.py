"""Logs per-request inference statistics to monitoring/inference_log.csv.

Only statistical features and an image hash are persisted — raw images are
never written to disk (GDPR-safe by design).
"""

from __future__ import annotations

import csv
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

LOG_PATH = Path("monitoring/inference_log.csv")

COLUMNS = [
    "timestamp",
    "image_hash",
    "pixel_mean_r",
    "pixel_mean_g",
    "pixel_mean_b",
    "pixel_std_r",
    "pixel_std_g",
    "pixel_std_b",
    "predicted_label",
    "confidence",
]

logger = logging.getLogger(__name__)


def log_inference(
    image_bytes: bytes,
    image: Image.Image,
    label: str,
    confidence: float,
) -> None:
    """Append one row of inference stats to the log. Never raises — a logging
    failure must never affect the prediction response."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        arr = np.array(image, dtype=np.float32)
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        image_hash = hashlib.md5(image_bytes).hexdigest()

        write_header = not LOG_PATH.exists()
        with LOG_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "image_hash": image_hash,
                    "pixel_mean_r": round(float(mean[0]), 4),
                    "pixel_mean_g": round(float(mean[1]), 4),
                    "pixel_mean_b": round(float(mean[2]), 4),
                    "pixel_std_r": round(float(std[0]), 4),
                    "pixel_std_g": round(float(std[1]), 4),
                    "pixel_std_b": round(float(std[2]), 4),
                    "predicted_label": label,
                    "confidence": round(confidence, 4),
                }
            )
    except Exception:
        logger.exception("inference_logger failed — prediction unaffected")
