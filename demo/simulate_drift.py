"""
Synthetic drift injection script — "Am I a Cat?" demo.

Generates out-of-distribution (OOD) images and:
  1. POSTs them to the FastAPI /predict endpoint to populate Prometheus metrics
  2. Writes synthetic inference log entries to monitoring/inference_log.csv
     with pixel statistics that diverge from the training baseline

Usage:
    python demo/simulate_drift.py [OPTIONS]

    --count       INT   Number of synthetic images to send (default: 50)
    --api-url     STR   FastAPI base URL (default: http://localhost:3000)
    --output-log  PATH  Path to inference_log.csv (default: monitoring/inference_log.csv)
    --seed        INT   Random seed for reproducibility (default: 42)
    --quiet             Suppress per-image progress output

Synthetic image types (equal distribution):
    noise     — uniform random RGB pixels  (very different from cat images)
    gradient  — smooth color gradient      (structured but OOD)
    solid     — single random flat colour  (zero variance per channel)
    blur      — heavily blurred cat image  (tests confidence degradation)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("simulate_drift")


# ---------------------------------------------------------------------------
# Image generators — all return PIL Image in RGB mode, 224×224
# ---------------------------------------------------------------------------

TARGET_SIZE = (224, 224)


def _noise_image(rng: random.Random) -> Image.Image:
    """Full-frame random noise — maximum distributional distance from natural images."""
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _gradient_image(rng: random.Random) -> Image.Image:
    """Smooth horizontal colour gradient — structured but out-of-distribution."""
    c1 = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
    c2 = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
    cols = np.linspace(0, 1, 224).reshape(1, 224, 1)
    arr = (c1 * (1 - cols) + c2 * cols).astype(np.uint8)
    arr = np.broadcast_to(arr, (224, 224, 3)).copy()
    return Image.fromarray(arr, mode="RGB")


def _solid_image(rng: random.Random) -> Image.Image:
    """Single flat colour — zero within-channel variance."""
    r, g, b = rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)
    arr = np.full((224, 224, 3), [r, g, b], dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _blur_image(rng: random.Random, source_dir: Path | None) -> Image.Image:
    """
    Heavily blurred version of a real image (if source_dir provided),
    or a blurred noise image (fallback). Models trained on sharp images
    often show confidence drops on blurry inputs.
    """
    if source_dir and source_dir.exists():
        candidates = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.jpeg"))
        if candidates:
            chosen = rng.choice(candidates)
            try:
                img = Image.open(chosen).convert("RGB").resize(TARGET_SIZE)
                return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(8, 18)))
            except Exception:
                pass
    # Fallback: blur a noise image
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=12))


# ---------------------------------------------------------------------------
# Inference log writer
# ---------------------------------------------------------------------------

LOG_COLUMNS = [
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


def _pixel_stats(img: Image.Image) -> dict:
    """Compute per-channel mean and std for an RGB PIL image."""
    arr = np.asarray(img).astype(np.float32)
    return {
        "pixel_mean_r": float(arr[:, :, 0].mean()),
        "pixel_mean_g": float(arr[:, :, 1].mean()),
        "pixel_mean_b": float(arr[:, :, 2].mean()),
        "pixel_std_r": float(arr[:, :, 0].std()),
        "pixel_std_g": float(arr[:, :, 1].std()),
        "pixel_std_b": float(arr[:, :, 2].std()),
    }


def _append_log_row(
    log_path: Path, img_bytes: bytes, label: str, confidence: float
) -> None:
    """Write one row to the inference log CSV (GDPR-safe — no raw image stored)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(TARGET_SIZE)
    stats = _pixel_stats(img)
    image_hash = hashlib.md5(img_bytes).hexdigest()

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "image_hash": image_hash,
        **stats,
        "predicted_label": label,
        "confidence": round(confidence, 6),
    }

    with log_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Core injection loop
# ---------------------------------------------------------------------------


def _pil_to_bytes(img: Image.Image) -> bytes:
    """Encode PIL image to JPEG bytes in memory."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def run_injection(
    count: int,
    api_url: str,
    output_log: Path,
    seed: int,
    quiet: bool,
    source_dir: Path | None,
) -> int:
    """
    Main injection function.  Returns the number of successfully processed images.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    generators = [_noise_image, _gradient_image, _solid_image]
    gen_names = ["noise", "gradient", "solid"]
    # Include blur only if we can generate images
    generators.append(lambda r: _blur_image(r, source_dir))  # type: ignore[arg-type]
    gen_names.append("blur")

    success_count = 0
    error_count = 0

    log.info("Starting drift injection: %d images → %s", count, api_url)
    log.info("Inference log → %s", output_log)

    with httpx.Client(timeout=15.0) as client:
        for i in range(count):
            # Round-robin through image types
            gen_idx = i % len(generators)
            gen_fn = generators[gen_idx]
            img_type = gen_names[gen_idx]

            try:
                img = gen_fn(rng)
                img_bytes = _pil_to_bytes(img)

                # 1. Send to API
                response = client.post(
                    f"{api_url}/predict",
                    files={
                        "file": (
                            f"synthetic_{img_type}_{i:04d}.jpg",
                            img_bytes,
                            "image/jpeg",
                        )
                    },
                )
                response.raise_for_status()
                result = response.json()
                label: str = result["label"]
                confidence: float = result["confidence"]

                # 2. Write to inference log
                _append_log_row(output_log, img_bytes, label, confidence)

                success_count += 1
                if not quiet:
                    log.info(
                        "[%3d/%d] type=%-8s  label=%-7s  confidence=%.4f",
                        i + 1,
                        count,
                        img_type,
                        label,
                        confidence,
                    )

            except httpx.HTTPStatusError as exc:
                error_count += 1
                log.warning(
                    "[%3d/%d] API error %d: %s",
                    i + 1,
                    count,
                    exc.response.status_code,
                    exc.response.text[:120],
                )
            except httpx.ConnectError:
                log.error("Cannot connect to API at %s — is it running?", api_url)
                return success_count
            except Exception as exc:
                error_count += 1
                log.warning("[%3d/%d] Unexpected error: %s", i + 1, count, exc)

            # Brief pause to avoid hammering the API
            time.sleep(0.05)

    log.info(
        "Injection complete: %d succeeded, %d errors. Log: %s",
        success_count,
        error_count,
        output_log,
    )
    return success_count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject synthetic OOD images into the Am I a Cat? API to simulate drift.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--count", type=int, default=50, help="Number of synthetic images"
    )
    parser.add_argument(
        "--api-url", default="http://localhost:3000", help="FastAPI base URL"
    )
    parser.add_argument(
        "--output-log",
        default="monitoring/inference_log.csv",
        help="Path to inference_log.csv",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-image log lines"
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional directory of real images for blur generation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    source = Path(args.source_dir) if args.source_dir else None
    n = run_injection(
        count=args.count,
        api_url=args.api_url,
        output_log=Path(args.output_log),
        seed=args.seed,
        quiet=args.quiet,
        source_dir=source,
    )
    sys.exit(0 if n > 0 else 1)
