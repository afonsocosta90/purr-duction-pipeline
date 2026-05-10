"""Drift detection using Evidently AI.

Compares the recent inference log against the training baseline and fires an
alert when distributions diverge or average confidence drops.

Usage:
    poetry run python -m src.catops.monitoring.drift
    make drift-report
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

from src.catops.monitoring.alerts import send_alert

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASELINE_PATH = Path("monitoring/baseline_stats.csv")
LOG_PATH = Path("monitoring/inference_log.csv")
REPORT_PATH = Path("monitoring/drift_report.html")
SCORE_PATH = Path("monitoring/drift_score.json")

FEATURE_COLS = [
    "pixel_mean_r",
    "pixel_mean_g",
    "pixel_mean_b",
    "pixel_std_r",
    "pixel_std_g",
    "pixel_std_b",
]

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "200"))
MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "50"))


def _load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not BASELINE_PATH.exists():
        logger.error("Baseline not found at %s — run 'make compute-baseline' first", BASELINE_PATH)
        sys.exit(1)

    if not LOG_PATH.exists():
        logger.error("Inference log not found at %s — no predictions have been logged yet", LOG_PATH)
        sys.exit(1)

    baseline = pd.read_csv(BASELINE_PATH)
    inference = pd.read_csv(LOG_PATH)

    if len(inference) < MIN_SAMPLES:
        logger.error(
            "Only %d rows in inference log — need at least %d for a reliable drift check",
            len(inference),
            MIN_SAMPLES,
        )
        sys.exit(1)

    recent = inference.tail(WINDOW_SIZE)
    logger.info("Baseline: %d rows · Inference window: %d rows", len(baseline), len(recent))
    return baseline, recent


def _run_evidently(baseline: pd.DataFrame, current: pd.DataFrame) -> tuple[float, bool]:
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=baseline[FEATURE_COLS],
        current_data=current[FEATURE_COLS],
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))
    logger.info("Drift report saved to %s", REPORT_PATH)

    result = report.as_dict()
    # DataDriftPreset puts DatasetDriftMetric first
    dataset_result = result["metrics"][0]["result"]
    drift_share: float = dataset_result.get("share_of_drifted_columns", 0.0)
    drift_detected: bool = dataset_result.get("dataset_drift", drift_share >= DRIFT_THRESHOLD)
    return drift_share, drift_detected


def _update_api_gauge(score: float) -> None:
    api_url = os.getenv("API_URL", "http://localhost:3000")
    endpoint = f"{api_url}/internal/drift-score"
    payload = json.dumps({"score": score}).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            logger.info("Updated catops_drift_score gauge on running API")
    except Exception:
        logger.debug("API not reachable — drift score saved to file only")


def main() -> None:
    baseline, recent = _load_dataframes()

    drift_score, drift_detected = _run_evidently(baseline, recent)
    avg_confidence = float(recent["confidence"].mean())

    logger.info(
        "drift_score=%.3f (threshold=%.2f) · avg_confidence=%.3f (threshold=%.2f)",
        drift_score,
        DRIFT_THRESHOLD,
        avg_confidence,
        CONFIDENCE_THRESHOLD,
    )

    score_data = {
        "score": round(drift_score, 4),
        "avg_confidence": round(avg_confidence, 4),
        "drift_detected": drift_detected,
        "low_confidence": avg_confidence < CONFIDENCE_THRESHOLD,
        "window_size": len(recent),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    SCORE_PATH.write_text(json.dumps(score_data, indent=2))
    logger.info("Drift score saved to %s", SCORE_PATH)

    _update_api_gauge(drift_score)

    should_alert = drift_detected or avg_confidence < CONFIDENCE_THRESHOLD
    if should_alert:
        logger.warning("🚨 Alert condition met — firing alert")
        send_alert(
            drift_score=drift_score,
            avg_confidence=avg_confidence,
            drift_threshold=DRIFT_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
    else:
        logger.info("✅ No drift detected — model healthy")

    sys.exit(1 if should_alert else 0)


if __name__ == "__main__":
    main()
