"""Send drift alerts via Slack webhook.

Set SLACK_WEBHOOK_URL to enable. If the variable is absent the alert is
printed to stdout so the GitHub Actions log still captures it.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def send_alert(
    drift_score: float,
    avg_confidence: float,
    drift_threshold: float,
    confidence_threshold: float,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    reasons = []
    if drift_score >= drift_threshold:
        reasons.append(f"drift score {drift_score:.3f} ≥ threshold {drift_threshold}")
    if avg_confidence < confidence_threshold:
        reasons.append(
            f"avg confidence {avg_confidence:.3f} < threshold {confidence_threshold}"
        )

    message = (
        f"⚠️ *Drift detected — purr-duction-pipeline* ({timestamp})\n"
        f"Reason: {' · '.join(reasons)}\n"
        f"Report: `monitoring/drift_report.html`\n"
        f"Action: collect new labelled images → `poetry run dvc repro --force`"
    )

    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print(f"[ALERT — no SLACK_WEBHOOK_URL set]\n{message}")
        return

    payload = json.dumps({"text": message}).encode()
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                logger.warning("Slack webhook returned status %s", resp.status)
            else:
                logger.info("Drift alert sent to Slack")
    except Exception:
        logger.exception("Failed to send Slack alert")
