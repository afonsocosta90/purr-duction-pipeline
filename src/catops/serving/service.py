"""Production model serving for "Am I a Cat?" — FastAPI"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import json
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.catops.monitoring.inference_logger import log_inference
from src.catops.serving.model_utils import (
    CLASS_NAMES,
    build_inference_transform,
    load_serving_model,
)

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
Image.MAX_IMAGE_PIXELS = 4_000_000  # ~2000×2000; guards against decompression bombs

ALLOWED_CONTENT_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif"}
)

_state: dict[str, Any] = {}

_PREDICTIONS = Counter(
    "catops_predictions_total",
    "Prediction count by label",
    ["label"],
)
_CONFIDENCE = Histogram(
    "catops_prediction_confidence",
    "Model confidence score distribution",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
)
_DRIFT_SCORE = Gauge(
    "catops_drift_score",
    "Latest Evidently AI drift score (share of drifted columns, 0–1; -1 = not yet computed)",
)


_DRIFT_SCORE_PATH = Path("monitoring/drift_score.json")


def _load_model_into_state() -> str:
    """(Re)load the serving model into the shared `_state` dict.

    Returns the load source ("registry" or "disk"). Safe to call while the
    server is live — see the /internal/reload-model endpoint for the
    concurrency reasoning.
    """
    model, source = load_serving_model()
    _state["model"] = model
    _state["transform"] = build_inference_transform()
    _state["device"] = next(model.parameters()).device
    return source


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    source = _load_model_into_state()
    logger.info("Model loaded from %s on device=%s", source, _state["device"])

    # Restore drift score from last drift run if available
    _DRIFT_SCORE.set(-1.0)
    if _DRIFT_SCORE_PATH.exists():
        try:
            data = json.loads(_DRIFT_SCORE_PATH.read_text())
            _DRIFT_SCORE.set(data["score"])
            logger.info(
                "Restored drift score=%.4f from %s", data["score"], _DRIFT_SCORE_PATH
            )
        except Exception:
            logger.warning("Could not read drift_score.json — gauge stays at -1")

    yield
    _state.clear()


app = FastAPI(
    title="Am I a Cat? API",
    description="Binary image classifier (Cat vs Not Cat)",
    version="1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


class Prediction(BaseModel):
    label: str
    confidence: float
    is_cat: bool


class _DriftScorePayload(BaseModel):
    score: float


@app.post("/internal/drift-score", include_in_schema=False)
async def update_drift_score(payload: _DriftScorePayload):
    """Called by drift.py after each drift check to keep the Prometheus gauge current."""
    _DRIFT_SCORE.set(payload.score)
    logger.info("catops_drift_score updated to %.4f", payload.score)
    return {"status": "ok", "score": payload.score}


@app.post("/internal/reload-model", include_in_schema=False)
async def reload_model():
    """Re-pull the serving model — called by the demo retrain flow so a newly
    registered @staging model is served without an API restart.

    No lock needed: this and /predict are both async coroutines on the single
    event loop, and /predict performs inference with no `await` after its
    initial `file.read()`, so a reload cannot interleave mid-prediction. The
    single uvicorn worker (see docker/Dockerfile) makes this the only process.
    """
    try:
        source = _load_model_into_state()
        logger.info("Model reloaded from %s", source)
        return {"status": "ok", "source": source}
    except Exception as exc:
        logger.exception("reload-model failed")
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc


@app.get("/health")
async def health():
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "cat-classifier"}


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    t0 = time.perf_counter()

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Accepted: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Could not decode image")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}") from exc

    device = _state["device"]
    input_tensor = _state["transform"](image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = _state["model"](input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        confidence, predicted_class = torch.max(probs, 0)

    label = CLASS_NAMES[predicted_class.item()]
    result = Prediction(
        label=label,
        confidence=round(float(confidence), 4),
        is_cat=(predicted_class.item() == 0),
    )

    _PREDICTIONS.labels(label=result.label).inc()
    _CONFIDENCE.observe(result.confidence)
    log_inference(data, image, result.label, result.confidence)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "predict label=%s confidence=%.4f latency_ms=%.1f filename=%s",
        result.label,
        result.confidence,
        latency_ms,
        file.filename or "<unnamed>",
    )
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
