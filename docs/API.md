# API Documentation – "Am I a Cat?"

**FastAPI serving layer** for the binary image classifier (Cat vs Not Cat).

**Source**: [`src/catops/serving/service.py`](../src/catops/serving/service.py)  
**Port**: `3000`  
**Auto-generated docs**: `http://localhost:3000/docs` (Swagger UI) · `http://localhost:3000/redoc`

---

## Running the API

### Locally (Poetry)

```bash
make serve
# equivalent: poetry run python -m src.catops.serving.service
```

The server starts on `http://localhost:3000`.

> The server requires `models/best_model.pt`, `models/model_config.json`, and  
> `data/processed/features_config.json` to be present. Run `make pipeline` or `poetry run dvc pull` first.

### Docker

```bash
# Build
docker build -f docker/Dockerfile -t purr-duction-pipeline .

# Run
docker run -p 3000:3000 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data/processed:/app/data/processed" \
  purr-duction-pipeline
```

The Docker image runs as non-root user (`appuser`, uid 1001) with 2 uvicorn workers.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_PATH` | `models/best_model.pt` | Path to the trained model state dict |
| `MODEL_CONFIG_PATH` | `models/model_config.json` | Path to model architecture config (num_classes, dropout) |
| `FEATURES_CONFIG_PATH` | `data/processed/features_config.json` | Path to preprocessing config (resize target, normalisation stats) |

---

## Endpoints

### `GET /health`

Returns the API liveness status. Returns `503` if the model failed to load at startup.

**Response `200`**

```json
{
  "status": "healthy",
  "model": "cat-classifier"
}
```

**Response `503`**

```json
{
  "detail": "Model not loaded"
}
```

**Example**

```bash
curl http://localhost:3000/health
```

---

### `POST /predict`

Classifies an uploaded image as `cat` or `not_cat`.

**Request**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | multipart/form-data | Yes | Image file to classify |

Accepted content types: `image/jpeg`, `image/png`, `image/webp`, `image/gif`  
Maximum file size: **10 MB**

**Response `200`**

```json
{
  "label": "cat",
  "confidence": 0.9871,
  "is_cat": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `label` | `"cat"` \| `"not_cat"` | Predicted class |
| `confidence` | `float` (0–1) | Softmax probability for the predicted class, rounded to 4 decimal places |
| `is_cat` | `bool` | Convenience flag: `true` when `label == "cat"` |

**Examples**

```bash
# Using curl
curl -X POST http://localhost:3000/predict \
  -F "file=@/path/to/image.jpg" | jq

# Using make (predicts a sample image from the processed dataset)
make api-test

# Using Python requests
import requests

with open("my_cat.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:3000/predict",
        files={"file": ("my_cat.jpg", f, "image/jpeg")},
    )
print(response.json())
# → {"label": "cat", "confidence": 0.9871, "is_cat": true}
```

**Error responses**

| Status | Condition |
|--------|-----------|
| `413` | File exceeds 10 MB |
| `415` | Content-type is not an accepted image MIME type |
| `422` | File could not be decoded as an image |
| `503` | Model is not loaded (startup failure) |

---

### `GET /metrics`

Exposes Prometheus metrics in text format. Scraped automatically by a Prometheus server.

```bash
curl http://localhost:3000/metrics
```

**Custom metrics**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `catops_predictions_total` | Counter | `label` (`cat` / `not_cat`) | Total predictions per class |
| `catops_prediction_confidence` | Histogram | — | Confidence score distribution; buckets at 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0 |

**Auto-instrumented HTTP metrics** (via `prometheus-fastapi-instrumentator`)

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests by method, path, status |
| `http_request_duration_seconds` | Histogram | Request latency distribution |

---

## How the Model is Loaded

At startup (`lifespan` context manager):

1. `load_model()` — reads `model_config.json` for architecture params (`num_classes`, `dropout`), rebuilds the ResNet50 head, loads the state dict with `weights_only=True`
2. `build_inference_transform()` — reads `features_config.json` for the resize target and normalisation stats used during training, so the serving transform always stays in sync with the pipeline
3. The model device (MPS on Apple Silicon, else CPU) is captured and reused for every request

---

## Inference Pipeline (per request)

```
Upload → content-type check → size check → PIL decode → decompression-bomb guard
→ transform (resize 224×224 · ToTensor · Normalize) → unsqueeze → model.forward()
→ softmax → argmax → Prediction response + Prometheus counters
```

The model runs in `torch.no_grad()` for performance. Latency (ms) is logged at `INFO` level for every prediction.

---

## Security Controls

| Control | Detail |
|---------|--------|
| File size limit | Reads `MAX + 1` bytes; raises `413` if exceeded — avoids loading oversized files into memory |
| Content-type validation | `415` returned before any decoding attempt if MIME type is not in the allowed set |
| Decompression bomb guard | `Image.MAX_IMAGE_PIXELS = 4_000_000` (~2000×2000); Pillow raises `DecompressionBombError` for larger images |
| Non-root Docker user | Container runs as `appuser` (uid 1001) |
| `weights_only=True` | `torch.load` uses safe deserialisation; prevents arbitrary code execution via pickle |
