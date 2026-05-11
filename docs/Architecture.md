# Architecture Documentation – "Am I a Cat?"

**Binary Image Classification MLOps System**  
*Cat vs. Not-Cat* – Production-Ready, Fully Automated, Reproducible Pipeline

**Version**: 2.0  
**Last Updated**: 2026-05-11  
**Status**: All 9 phases complete — full closed-loop MLOps system with live demo stack

## 1. Project Overview

The “Am I a Cat?” system is a **binary image classifier** that determines whether an input image contains a cat or not.  
What starts as a simple notebook model will be transformed into a **complete MLOps platform** with:
- Automated, versioned data pipelines
- Reproducible experiments with full lineage
- Continuous training & model promotion
- Production-grade serving with monitoring & drift detection
- Zero-downtime CI/CD deployments

**Key Objectives**
- 100% reproducibility (code + data + config + environment)
- Automated quality gates at every stage
- Model governance & promotion rules
- Scalable serving (batch + real-time)
- Observability & alerting for data/model drift

## 2. Phase Roadmap

| Phase | Description                                      | Status    |
|-------|--------------------------------------------------|-----------|
| 1     | Project scaffold (Poetry + DVC + Git)            | ✅ Done   |
| 2     | Data ingestion + validation                      | ✅ Done   |
| 3     | Feature engineering + stratified train/val/test split | ✅ Done   |
| 4     | Model training (PyTorch + Hydra + MLflow)        | ✅ Done   |
| 5     | Experiment tracking (MLflow full integration)    | ✅ Done   |
| 6     | CI/CD pipeline (GitHub Actions)                  | ✅ Done   |
| 7     | Model serving (FastAPI + Prometheus)             | ✅ Done   |
| 8     | Monitoring + drift detection                     | ✅ Done   |
| 9     | Interactive demo & portfolio showcase            | ✅ Done   |

---

## 3. High-Level Architecture

```mermaid
flowchart TD
    subgraph Data["Data Layer ✅ Phases 1-2"]
        RAW["data/raw/images/<br/>Uppercase first letter → cat<br/>lowercase → not_cat"]
        INGEST["ingest.py<br/>classify by filename · copy to processed/"]
        PROC["data/processed/<br/>cat/ · not_cat/ · metadata.csv"]
        VAL["validate.py<br/>quality gates"]
    end
    subgraph Features["Feature Layer ✅ Phase 3"]
        SPLIT["stratified split<br/>train / val / test"]
        FEAT["resize · normalize · augment"]
    end
    subgraph Training["Training + Evaluation ✅ Phases 4–5"]
        HYDRA["Hydra config"]
        DS["CatDataset<br/>split-aware CSV loader"]
        TRAIN["PyTorch trainer<br/>ResNet50 transfer learning"]
        EVAL["evaluate.py<br/>acc · F1 · precision · recall · ROC-AUC"]
        MLFLOW["MLflow<br/>metrics + artifacts + model promotion"]
    end
    subgraph CICD["CI/CD ✅ Phase 6"]
        GH["GitHub Actions<br/>lint → test → dvc repro → artifacts → docker"]
    end
    subgraph Serving["Serving ✅ Phase 7"]
        UTILS["model_utils.py<br/>load_model · build_inference_transform"]
        API["FastAPI service.py<br/>/predict · /health · /metrics"]
    end
    subgraph Monitor["Monitoring ✅ Phase 8"]
        LOG["inference_logger.py<br/>save image stats per request"]
        DRIFT["drift.py<br/>Evidently AI report"]
        PROM2["Prometheus + Grafana<br/>confidence · label dist · latency"]
        ALERT["alerts.py<br/>Slack / email on drift"]
    end
    subgraph Demo["Demo ✅ Phase 9"]
        STREAMLIT["streamlit_app.py<br/>drag-and-drop predict · feedback"]
        SIMDRIFT["simulate_drift.py<br/>synthetic drift injection"]
        RETRAIN["pipeline control<br/>dvc repro streaming · before/after metrics"]
    end
    RAW --> INGEST --> PROC --> VAL
    VAL --> SPLIT --> FEAT --> TRAIN
    HYDRA --> TRAIN
    DS --> TRAIN --> EVAL --> MLFLOW
    MLFLOW --> GH --> UTILS --> API
    API -->|"log image stats"| LOG
    LOG --> DRIFT --> ALERT
    PROM2 -->|"confidence drops"| ALERT
    ALERT -->|"human adds labelled data"| RAW
    STREAMLIT -->|"POST /predict"| API
    SIMDRIFT -->|"POST /predict (noise)"| API
    RETRAIN -->|"dvc repro --force"| TRAIN
```

---

## 4. Sequence Diagrams

### 4.1 Data Pipeline (Phases 1–2)

```mermaid
sequenceDiagram
    actor Dev as Developer
    participant FS as Filesystem
    participant DVC
    participant Ingest as ingest.py
    participant Validate as validate.py
    participant Remote as DVC Remote

    Note over Dev,FS: Naming rule — Uppercase first letter → cat · lowercase → not_cat
    Dev->>FS: place named images in data/raw/images/<br/>(e.g. Siamese_01.jpg → cat · labrador_01.jpg → not_cat)
    Dev->>DVC: dvc repro
    DVC->>Ingest: run stage: ingest
    Ingest->>FS: scan data/raw/images/ for *.jpg / *.jpeg / *.png
    Ingest->>FS: img.name[0].isupper() → copy to data/processed/cat/
    Ingest->>FS: img.name[0].islower() → copy to data/processed/not_cat/
    Ingest-->>DVC: stage complete (7,390 images)
    DVC->>Validate: run stage: validate
    Validate->>FS: scan data/processed/ → build metadata.csv
    Validate->>Validate: gate 1 — both classes present?
    Validate->>Validate: gate 2 — each class ≥ 30% of total?
    Validate->>Validate: gate 3 — no file smaller than 1 KB?
    Validate->>Validate: gate 4 — no duplicate image paths?
    Validate->>FS: write data/processed/metadata.csv
    Validate-->>DVC: PASSED ✅ (raises RuntimeError on failure ❌)
    DVC->>FS: write dvc.lock (content hashes)
    Dev->>DVC: dvc push
    DVC->>Remote: upload data/processed/ artifacts
    Dev->>FS: git commit dvc.lock && git push
```

### 4.2 Feature Engineering (Phase 3)

```mermaid
sequenceDiagram
    participant DVC
    participant BF as build_features.py
    participant FS as Filesystem

    DVC->>BF: run stage: features (depends on validate)
    BF->>FS: read data/processed/metadata.csv
    BF->>BF: stratified split 70/15/15 (seed=42, stratify=label)
    BF->>FS: write data/processed/train.csv · val.csv · test.csv
    BF->>BF: compute resize target (224×224) + normalisation stats
    BF->>FS: write data/processed/features_config.json
    BF-->>DVC: stage complete
```

### 4.3 Training & Evaluation (Phases 4–5)

```mermaid
sequenceDiagram
    participant DVC
    participant Hydra
    participant DS as CatDataset
    participant Train as train.py
    participant Eval as evaluate.py
    participant MLflow as MLflow (DagsHub)

    DVC->>Train: run stage: train (depends on features)
    Train->>Hydra: load configs/config.yaml (model + training)
    Train->>DS: CatDataset(train.csv, transform=build_inference_transform())
    DS->>DS: label mapping: cat→1 · not_cat→0
    Train->>Train: ResNet50 (IMAGENET1K_V1 weights) + Adam + CrossEntropyLoss
    Train->>MLflow: log params · per-epoch train loss
    loop Each epoch
        Train->>DS: DataLoader batch (train split)
        Train->>Train: forward · loss · backward · step
        Train->>DS: DataLoader batch (val split)
        Train->>Eval: compute accuracy · F1 · precision · recall · ROC-AUC
        Eval-->>Train: val metrics
        Train->>MLflow: log val metrics
    end
    Train->>Train: promotion gate — accuracy ≥ 0.94 AND F1 ≥ 0.93?
    alt promoted
        Train->>MLflow: log model to registry + tag "staging"
    end
    Eval->>MLflow: log confusion_matrix.png · roc_curve.png · classification_report.txt
    Train->>DVC: output models/best_model.pt
```

### 4.4 CI/CD (Phase 6)

```mermaid
sequenceDiagram
    actor Dev as Developer
    participant GH as GitHub Actions
    participant DVC
    participant Remote as DVC Remote (DagsHub)
    participant GHCR as GitHub Container Registry

    Dev->>GH: git push → trigger workflow
    GH->>GH: job: quality — make lint + make test
    GH->>Remote: dvc pull (restore versioned data)
    GH->>DVC: make pipeline (dvc repro)
    DVC-->>GH: models/best_model.pt + artifacts/
    GH->>GH: upload artifacts (GitHub Actions artifact store)
    Note over GH,GHCR: docker job runs on workflow_dispatch only
    GH->>GHCR: docker buildx build & push ghcr.io/<owner>/purr-duction-pipeline:latest
```

### 4.5 Inference / Serving (Phase 7)

```mermaid
sequenceDiagram
    actor Client
    participant API as FastAPI service.py
    participant Utils as model_utils.py
    participant Prom as Prometheus

    Note over API,Utils: Startup (lifespan)
    API->>Utils: load_model() — reads MODEL_PATH env var
    Utils->>Utils: build ResNet50 from model_config.json · load state dict
    API->>Utils: build_inference_transform() — reads FEATURES_CONFIG_PATH
    Utils->>Utils: resize 224×224 · normalise from features_config.json

    Client->>API: GET /health
    API-->>Client: 200 OK {status: healthy} · 503 if model not loaded

    Client->>API: POST /predict (multipart image)
    API->>API: content-type check → 415 if non-image
    API->>API: size check → 413 if > 10 MB
    API->>API: decompression-bomb guard (MAX_IMAGE_PIXELS = 4 000 000)
    API->>Utils: apply transform · model.forward()
    Utils-->>API: softmax probabilities
    API->>Prom: catops_predictions_total{label} += 1
    API->>Prom: catops_prediction_confidence.observe(confidence)
    API-->>Client: {label, confidence, is_cat}

    Client->>API: GET /metrics
    API-->>Client: Prometheus text format (HTTP + custom metrics)
```

### 4.6 Monitoring & Drift Detection (Phase 8)

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI service.py
    participant Logger as inference_logger.py
    participant Log as monitoring/inference_log.csv
    participant GH as GitHub Actions (drift.yml · daily 08:00 UTC)
    participant Drift as drift.py (Evidently AI)
    participant Baseline as monitoring/baseline_stats.csv
    participant Prom as Prometheus gauge (catops_drift_score)
    participant Grafana
    participant Alert as alerts.py
    actor Dev as Developer

    User->>API: POST /predict (image upload)
    API->>API: classify image → {label, confidence}
    API->>Logger: log_inference(image_bytes, label, confidence)
    Logger->>Logger: hash = MD5(image_bytes) · compute pixel mean/std per channel
    Logger->>Log: append row (hash, pixel_mean/std_r/g/b, label, confidence, timestamp)
    Note over Logger: raw image never stored — GDPR-safe
    API-->>User: {label, confidence, is_cat}

    Note over GH,Baseline: Runs daily at 08:00 UTC via GitHub Actions scheduled workflow
    GH->>Drift: trigger drift.py
    Drift->>Log: read last N rows from inference_log.csv
    Drift->>Baseline: read monitoring/baseline_stats.csv (DVC-tracked, versioned with model)
    Drift->>Drift: Evidently DataDriftPreset — column-level drift on pixel mean/std
    Drift->>Drift: compute rolling average confidence
    Drift->>Prom: update catops_drift_score gauge
    Prom->>Grafana: drift score visible on dashboard automatically
    alt drift detected OR avg confidence < 0.80
        Drift->>Alert: trigger alert
        Alert->>Dev: Slack webhook — drift score · confidence · link to drift_report.html
        Dev->>Dev: collect new labelled images<br/>(Uppercase=cat · lowercase=not_cat)
        Dev->>Dev: drop into data/raw/images/ → dvc repro --force → retrain
    end
```

### 4.7 Interactive Demo (Phase 9)

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI<br/>(demo/streamlit_app.py)
    participant API as FastAPI :3000
    participant Sim as simulate_drift.py
    participant DVC as dvc repro --force
    participant Prom as Prometheus :9090
    participant Grafana as Grafana :3001

    Note over UI,Grafana: `make demo` brings up all four services via docker-compose

    User->>UI: upload image (Live Prediction tab)
    UI->>API: POST /predict (multipart)
    API-->>UI: {label, confidence, is_cat}
    UI-->>User: prediction card + confidence gauge
    User->>UI: submit correct label (feedback radio)
    UI->>UI: append row to monitoring/feedback_log.csv

    User->>UI: Pipeline Control tab → Inject Drift
    UI->>Sim: spawn simulate_drift.py --count N
    Sim->>API: POST /predict × N (random noise / gradient / solid-colour / blurred images)
    API->>API: log pixel stats to monitoring/inference_log.csv

    User->>UI: Retrain Model button
    UI->>DVC: subprocess dvc repro --force (ingest → validate → features → train → evaluate)
    DVC-->>UI: stdout streamed live in log box
    UI-->>User: before/after metrics table + promotion verdict

    Prom->>Grafana: scrape :3000/metrics (predictions_total · confidence · latency)
    User->>Grafana: open :3001 → pre-provisioned CatOps dashboard
```

---

## 5. Detailed Component Architecture

### 5.1 Data Layer (✅ implemented)

**ingest.py** (`src/catops/data/ingest.py`)

| Responsibility | Detail |
|----------------|--------|
| Source dataset | Oxford-IIIT Pet (37 breeds, JPEG); any user-added images following the same convention |
| Classification rule | `filename[0].isupper()` → `cat`; `filename[0].islower()` → `not_cat` |
| Input | `data/raw/images/` — flat directory; optionally bootstrapped from `images.tar.gz` |
| Output | `data/processed/cat/` and `data/processed/not_cat/` |
| Idempotent | Skips tar extraction if `data/raw/images/` already exists |
| Adding new data | Place new files in `data/raw/images/` with correct capitalisation (see `docs/HowToAddData.md`) |

**validate.py** (`src/catops/data/validate.py`)

| Quality Gate | Rule |
|--------------|------|
| Class existence | Both `cat` and `not_cat` must be present with > 0 images |
| Class balance | Each class must be ≥ 30% of total |
| File integrity | No files smaller than 1 KB |
| Deduplication | No duplicate `image_path` values |
| Output | `data/processed/metadata.csv` (columns: image_path, label, file_size) |

**DVC pipeline** (`dvc.yaml`)
```
ingest  ──►  validate
  │               │
  └─► data/processed/   (shared, DVC-tracked)
```
Both stages are cached — `dvc repro` is a no-op if inputs are unchanged.

### 5.1.3 Features Stage (Phase 3 – ✅ Done)

**build_features.py** (`src/catops/features/build_features.py`)

| Responsibility                  | Detail |
|---------------------------------|--------|
| Stratified split                | 70/15/15 using `label` column + fixed seed=42 |
| Outputs                         | `train.csv`, `val.csv`, `test.csv` (paths + labels only) |
| Preprocessing decisions         | Resize target (224×224) + normalization stats (saved as `features_config.json`) |
| Validation                      | Extended gates in `validate.py` (class balance preserved, no leakage) |
| DVC stage                       | `features` (cached, depends on `validate`) |

All artifacts are now part of the official **data contract** consumed by training.

### 5.2 Training Layer (Phases 4–5 — ✅ implemented)

**dataset.py** (`src/catops/data/dataset.py`)

| Responsibility | Detail |
|---|---|
| Purpose | Split-aware image loader — replaces `ImageFolder` over raw dirs |
| Input | Any of `train.csv`, `val.csv`, `test.csv` (columns: `image_path`, `label`) |
| Label mapping | `cat → 1`, `not_cat → 0` |
| Transforms | Injected at construction (resize + normalize from `features_config.json`) |

**train.py** (`src/catops/training/train.py`)

| Responsibility | Detail |
|---|---|
| Model | ResNet50 — modern `ResNet50_Weights.IMAGENET1K_V1` API (no deprecation warnings) |
| Dataset | `CatDataset(train.csv)` — respects stratified split, no data leakage |
| Device | MPS (Apple Silicon) if available, else CPU |
| Optimizer | Adam, lr from `cfg.training.learning_rate` |
| Loss | CrossEntropyLoss |
| Config | Hydra (`configs/config.yaml` → `model.yaml` + `training.yaml`) |
| Reproducibility | `torch.manual_seed` fixed via `cfg.training.seed` |
| Tracking | MLflow experiment `am-i-a-cat`: params, per-epoch train loss, val metrics, artifacts |
| Promotion | Real val metrics gated: accuracy ≥ 0.94 **and** F1 ≥ 0.93 → logged to MLflow registry + tagged `staging` |
| DVC output | `models/best_model.pt` (state dict) |

**evaluate.py** (`src/catops/evaluation/evaluate.py`)

| Metric / Artifact | Detail |
|---|---|
| Accuracy | `sklearn.metrics.accuracy_score` on val (or test) split |
| F1 | Weighted F1 score |
| Precision / Recall | Weighted, per-class breakdown |
| ROC-AUC | Binary, on softmax probabilities |
| Confusion matrix | Seaborn heatmap — logged as `artifacts/confusion_matrix.png` |
| ROC curve | Matplotlib — logged as `artifacts/roc_curve.png` |
| Classification report | Text artifact — logged as `artifacts/classification_report.txt` |

**Configs** (`configs/`)

| File | Key settings |
|---|---|
| `model.yaml` | `name: resnet50`, `pretrained: true`, `num_classes: 2`, `dropout: 0.2` |
| `training.yaml` | `epochs: 10`, `batch_size: 32`, `lr: 0.001`, `seed: 42` |
| `training.yaml` (promotion) | `min_accuracy: 0.94`, `min_f1: 0.93`, `min_precision: 0.90` |

**DVC train stage dependencies (updated)**
```
features → train (depends on: train.py, evaluate.py, dataset.py, feature CSVs, configs)
           outs:  models/best_model.pt, artifacts/
```

### 5.4 Serving Layer (Phase 7 — ✅ implemented)

**model_utils.py** (`src/catops/serving/model_utils.py`)

| Responsibility | Detail |
|---|---|
| `load_model()` | Builds ResNet50 architecture from `model_config.json`, loads state dict with `weights_only=True`; path overridable via `MODEL_PATH` env var |
| `build_inference_transform()` | Reads resize target + normalization stats from `features_config.json` at runtime; path overridable via `FEATURES_CONFIG_PATH` env var |
| Class mapping | `CLASS_NAMES = {0: "cat", 1: "not_cat"}` — matches `CatDataset` encoding |

**service.py** (`src/catops/serving/service.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns 200 if model is loaded, 503 otherwise |
| `/predict` | POST (multipart) | Accepts image upload, returns `{label, confidence, is_cat}` |
| `/metrics` | GET | Prometheus text format — HTTP instrumentation + custom metrics |

| Security / robustness feature | Detail |
|---|---|
| File size limit | 10 MB max; reads MAX+1 bytes to detect without loading full file |
| Decompression bomb guard | `Image.MAX_IMAGE_PIXELS = 4_000_000` (~2000×2000) |
| Content-type pre-validation | 415 for non-image MIME types before any decode attempt |
| Device handling | Input tensor moved to model device with `.to(device)` |

| Prometheus metric | Type | Labels |
|---|---|---|
| `catops_predictions_total` | Counter | `label` (cat / not_cat) |
| `catops_prediction_confidence` | Histogram | — (buckets 0.5–1.0) |
| HTTP request metrics | Counter + Histogram | auto-instrumented via `prometheus-fastapi-instrumentator` |

**Docker**

| Setting | Value |
|---|---|
| Base image | `python:3.12-slim` |
| User | `appuser` (uid 1001) — non-root |
| Port | 3000 |
| CMD | `uvicorn src.catops.serving.service:app --host 0.0.0.0 --port 3000 --workers 2 --access-log` |

### 5.5 CI/CD (Phase 6 — ✅ implemented)

**`ci-cd.yml`** — triggers on every push/PR to `main` and `workflow_dispatch`:

| Job | Steps | Condition |
|-----|-------|-----------|
| `quality` | Install deps → `make lint` → `make test` | always |
| `pipeline` | Install deps → configure DVC remote (DagsHub) → `dvc pull` → `make pipeline` → upload `models/best_model.pt` + `artifacts/` | after `quality`, code changed only |
| `docker` | Docker Buildx → GHCR login → build & push `ghcr.io/<owner>/purr-duction-pipeline:latest` | `workflow_dispatch` on `main` only |

**`drift.yml`** (Phase 8) — triggers on `schedule: cron '0 8 * * *'` and `workflow_dispatch`:

| Job | Steps |
|-----|-------|
| `drift` | Install deps → `make drift-report` → upload `monitoring/drift_report.html` as artifact → alert if drift detected |

**Secrets required**: `DAGSHUB_USER`, `DAGSHUB_TOKEN` (DVC remote + MLflow auth) · `SLACK_WEBHOOK_URL` (Phase 8 alerts).

**MLflow environment variables** (set automatically in the `pipeline` job):
- `MLFLOW_TRACKING_URI` → `https://dagshub.com/afonsocosta90/purr-duction-pipeline.mlflow`
- `MLFLOW_TRACKING_USERNAME` → `DAGSHUB_USER` secret
- `MLFLOW_TRACKING_PASSWORD` → `DAGSHUB_TOKEN` secret

For local development, export these or omit to use the DagsHub URI unauthenticated (read-only public).

### 5.6 Monitoring & Drift Detection (Phase 8 — ✅ implemented)

**The core problem**: once the model is deployed, the images people send to `/predict` may gradually look different from the training data. The model silently degrades — confidence drops, misclassifications increase — with no visible signal unless you instrument it.

**Solution**: log a statistical fingerprint of every inference request, periodically compare it against the training baseline, and alert when the distributions diverge.

---

**inference_logger.py** (`src/catops/monitoring/inference_logger.py`)

Called inside `/predict` after every successful classification — adds one row to the inference log without blocking the response.

| Field logged | Detail |
|---|---|
| `timestamp` | ISO-8601 request time |
| `image_hash` | MD5 of raw image bytes — deduplication + GDPR-safe identifier (no raw image stored) |
| `pixel_mean_r/g/b` | Per-channel mean pixel value of the uploaded image |
| `pixel_std_r/g/b` | Per-channel standard deviation |
| `predicted_label` | `cat` or `not_cat` |
| `confidence` | Softmax score for the predicted class |

Output: `monitoring/inference_log.csv` — append-only, one row per request.

**Privacy by design**: only statistical features and a hash are stored. Raw images are never written to disk — compliant with GDPR without any extra effort.

---

**drift.py** (`src/catops/monitoring/drift.py`)

Triggered automatically by a **GitHub Actions scheduled workflow** (`drift.yml`, daily at 08:00 UTC). Can also be run manually with `make drift-report`.

| Step | Detail |
|---|---|
| Load baseline | Read `monitoring/baseline_stats.csv` (pre-computed from training images, DVC-tracked) |
| Load current window | Read last N rows from `monitoring/inference_log.csv` |
| Evidently `DataDriftPreset` | Column-level drift test on pixel mean/std channels |
| Output | `monitoring/drift_report.html` — human-readable report |
| Drift score | Extracted from report; exposed as `catops_drift_score` Prometheus gauge so Grafana picks it up automatically |
| Confidence check | Rolling average confidence of last N requests compared against threshold (0.80) |

---

**alerts.py** (`src/catops/monitoring/alerts.py`)

Fires when `drift.py` detects drift or when average confidence in the inference window drops below the threshold.

| Condition | Trigger |
|---|---|
| Evidently drift score > threshold | Distribution of incoming images differs from training |
| Rolling average confidence < 0.80 | Model is uncertain — likely out-of-distribution inputs |

Alert payload: drift score, confidence trend, link to `drift_report.html`, instructions to retrain.

---

**Grafana dashboard** (`docker-compose.yml` + `monitoring/grafana/dashboard.json`)

All panels are fully automated — no manual data push needed.

| Panel | Source metric |
|---|---|
| Prediction label distribution over time | `catops_predictions_total` (Prometheus counter) |
| Confidence histogram | `catops_prediction_confidence` (Prometheus histogram) |
| Request rate | `http_requests_total` (Prometheus counter, auto-instrumented) |
| Drift score trend | `catops_drift_score` (Prometheus gauge, updated by `drift.py`) |

---

**Retraining loop**

The monitoring system cannot label production images automatically — uploaded images have no ground-truth label. The loop is human-assisted:

```
Alert fires
    → Developer reviews drift_report.html
    → Collects new labelled images (Uppercase filename = cat, lowercase = not_cat)
    → Drops them into data/raw/images/
    → Runs: poetry run dvc repro --force && poetry run dvc push
    → New model trained, evaluated, and promoted if it passes the promotion gate
```

### 5.7 Interactive Demo Layer (Phase 9 — ✅ implemented)

A single `make demo` command spins up the complete local demo stack via `demo/docker-compose.demo.yml`.

| Service | Port | Purpose |
|---------|------|---------|
| Streamlit UI | 8501 | Drag-and-drop prediction, feedback submission, pipeline control |
| FastAPI | 3000 | Inference backend (same image as production) |
| Grafana | 3001 | Pre-provisioned CatOps dashboard (admin / catops) |
| Prometheus | 9090 | Metrics store scraped from API |

**streamlit_app.py** (`demo/streamlit_app.py`)

| Tab | Functionality |
|-----|---------------|
| Live Prediction | Upload any image → ResNet50 prediction card + confidence gauge; feedback radio saves to `monitoring/feedback_log.csv` |
| Pipeline Control | Inject synthetic drift (random noise / gradient / solid-colour / blurred) and trigger `dvc repro --force` with live log streaming + before/after metrics comparison |
| Monitoring | Live label distribution chart and rolling average confidence from `monitoring/inference_log.csv` |

**simulate_drift.py** (`demo/simulate_drift.py`)

Generates out-of-distribution images (random noise, gradient, solid-colour, blurred) and fires them at `/predict` to populate `monitoring/inference_log.csv` with drift signal for the Evidently AI report.

**Grafana dashboard** (`demo/grafana/` + `demo/prometheus.yml`)

Pre-provisioned *CatOps — Am I a Cat? Live Metrics* dashboard with:
- Total prediction counter + per-label breakdown
- Confidence gauge (avg) with red/yellow/green thresholds
- Time-series: prediction volume, confidence trend, request latency percentiles (p50/p95/p99)
- Label distribution bar chart + confidence histogram

**Demo commands**

| Command | Effect |
|---------|--------|
| `make demo` | Build and start all four services |
| `make demo-logs` | Follow logs from all services |
| `make demo-down` | Stop the stack |
| `make demo-reset` | Hard reset — remove volumes + images, rebuild from scratch |
| `DOCKER_HOST=ssh://user@host make demo-cloud` | Deploy to a remote server |

## 6. Technology Stack

| Layer               | Tool                          | Status   | Reason |
|---------------------|-------------------------------|----------|--------|
| Dependency mgmt     | Poetry                        | ✅ Active | Reproducible lockfile |
| Data versioning     | DVC + local / S3 / GCS remote | ✅ Active | Git-like semantics for large files |
| Code quality        | Ruff + Black + Mypy           | ✅ Active | Lint, format, type-check |
| Pre-commit hooks    | pre-commit + DVC hooks        | ✅ Active | Enforce DVC sync on commit/push |
| Data processing     | Pillow + pandas + tqdm        | ✅ Active | Image handling and metadata |
| Config management   | Hydra + OmegaConf             | ✅ Active | Multi-run, sweepable configs |
| Training            | PyTorch 2 + TorchVision       | ✅ Active | ResNet50 transfer learning, split-aware CatDataset |
| Evaluation          | scikit-learn + matplotlib + seaborn | ✅ Active | Full classification metrics + confusion matrix + ROC curve |
| Experiment tracking | MLflow (DagsHub remote)       | ✅ Active | Run tracking, artifact logging, model promotion; tracking URI via `MLFLOW_TRACKING_URI` env var |
| Serving             | FastAPI + Prometheus          | ✅ Active | Async inference, Prometheus metrics, non-root Docker |
| Containerisation    | Docker                        | ✅ Active | Non-root user, uvicorn multi-worker CMD |
| CI/CD               | GitHub Actions                | ✅ Active | quality → pipeline → docker jobs; DagsHub DVC remote |
| Monitoring          | Evidently AI + Prometheus + Grafana | ✅ Active | Inference logging · drift reports · confidence dashboards · alerts |
| Demo UI             | Streamlit + Plotly + Docker Compose | ✅ Active | Closed-loop demo: predict · feedback · drift · retrain · before/after metrics |
| Cloud (optional)    | AWS / GCP (S3 + GPU runners)  | Planned  | Scalability |

---

## 7. Repository Layout

```
purr-duction-pipeline/
├── src/
│   └── catops/
│       ├── __init__.py
│       ├── data/
│       │   ├── ingest.py          # Phase 2: classify by filename capitalisation + copy to processed/
│       │   ├── validate.py        # Phase 2: quality gates + metadata.csv
│       │   └── dataset.py         # Phase 3: CatDataset, split-aware CSV loader
│       ├── features/
│       │   └── build_features.py  # Phase 3: stratified split + features_config.json
│       ├── training/
│       │   └── train.py           # Phases 4–5: ResNet50 + MLflow tracking + promotion gate
│       ├── evaluation/
│       │   └── evaluate.py        # Phase 5: accuracy · F1 · confusion matrix · ROC curve
│       ├── serving/               # Phase 7: FastAPI service
│       │   ├── model_utils.py     # load_model() + build_inference_transform()
│       │   └── service.py         # /predict · /health · /metrics
│       ├── monitoring/            # Phase 8: drift detection
│       │   ├── inference_logger.py  # log pixel stats + confidence per request
│       │   ├── drift.py             # Evidently AI drift report
│       │   └── alerts.py            # Slack / email notifications
│       └── utils/                 # shared logging, config helpers
├── configs/                       # Hydra configs: config.yaml → model.yaml + training.yaml
├── pipelines/                     # DVC stage helpers + future orchestration flows
├── data/
│   ├── raw/
│   │   └── images/                # flat directory — filenames determine class (Uppercase=cat)
│   └── processed/
│       ├── cat/                   # owned by ingest.py
│       ├── not_cat/               # owned by ingest.py
│       ├── metadata.csv           # owned by validate.py
│       ├── train.csv              # owned by build_features.py
│       ├── val.csv                # owned by build_features.py
│       ├── test.csv               # owned by build_features.py
│       └── features_config.json   # resize target + normalisation stats
├── models/
│   └── best_model.pt              # DVC output of train stage
├── artifacts/                     # confusion_matrix.png · roc_curve.png · classification_report.txt
├── monitoring/                    # Phase 8 runtime outputs
│   ├── inference_log.csv          # append-only log (one row per /predict call)
│   ├── feedback_log.csv           # user feedback from Streamlit demo
│   ├── baseline_stats.csv         # pixel stats of training set, DVC-tracked
│   ├── drift_report.html          # Evidently AI report (generated on demand)
│   ├── grafana/                   # Grafana provisioning configs
│   └── prometheus.yml             # Prometheus scrape config (targets: API :3000/metrics)
├── demo/                          # Phase 9: interactive demo stack
│   ├── streamlit_app.py           # Streamlit UI (predict · feedback · pipeline control · monitoring)
│   ├── simulate_drift.py          # synthetic drift injection — fires OOD images at /predict
│   ├── Dockerfile.demo            # demo-specific Dockerfile (Streamlit + httpx)
│   ├── docker-compose.demo.yml    # full 4-service stack: API + Streamlit + Prometheus + Grafana
│   ├── prometheus.yml             # Prometheus scrape config for demo stack
│   ├── grafana/                   # pre-provisioned CatOps dashboard
│   └── demo_data/                 # sample images for live demo
├── docker/                        # Dockerfile (non-root, uvicorn multi-worker)
├── docker-compose.yml             # base compose for API + Prometheus + Grafana
├── .github/workflows/             # ci-cd.yml: quality → pipeline → docker · drift.yml: daily 08:00 UTC
├── tests/                         # unit + integration + data validation tests
├── notebooks/                     # exploration only (never committed with outputs)
├── docs/
│   ├── Architecture.md
│   ├── API.md
│   ├── HowToAddData.md
│   └── ProjectScope.md
├── dvc.yaml                       # pipeline stage definitions
├── dvc.lock                       # content hashes (commit this, not the data)
├── params.yaml                    # experiment parameters
├── pyproject.toml                 # Poetry: deps + tool config
├── Makefile                       # developer commands (install · test · lint · pipeline · serve · demo)
├── .pre-commit-config.yaml
└── README.md
```