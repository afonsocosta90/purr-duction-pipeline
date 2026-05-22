# Am I a Cat? — MLOps Portfolio Project

**Binary image classification (Cat vs Not Cat) built as a complete, production-grade MLOps platform.**

ResNet50 transfer learning · DVC pipelines · MLflow experiment tracking · FastAPI serving · Evidently AI drift detection · Prometheus + Grafana observability · GitHub Actions CI/CD · Interactive Streamlit demo

> 7,390 training images · >94% val accuracy · 9 phases · all complete ✅

---

## Live Demo — One Command

Pick whichever fits your machine — all three do the same thing (fetch the model,
build/pull images, start the stack):

```bash
make demo                    # any OS with make (Windows, macOS, Linux)
python demo/launch.py        # any OS with Python — no make needed
```

```powershell
.\demo.cmd                   # Windows — no make, no Python knowledge needed
```

```bash
./demo.sh                    # macOS / Linux — no make needed
```

Builds and starts four Docker services:

| Service | URL | Notes |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Main demo — start here |
| FastAPI | http://localhost:3000/docs | Swagger / REST API |
| Grafana | http://localhost:3001 | Login: `admin` / `catops` |
| Prometheus | http://localhost:9090 | Raw metrics |

Stop or reset it the same way — `make demo-down` / `make demo-reset`, or
`python demo/launch.py down` / `reset` (also `down`/`reset` on the `demo.cmd` /
`demo.sh` wrappers).

### Requirements

- Docker Desktop installed and running
- ~4 GB disk space (PyTorch base image)
- Ports 3000, 3001, 8501, 9090 free
- Python 3 on PATH (used by the launcher; the project needs it anyway)

> The launcher is cross-platform — it runs the same from Windows PowerShell/cmd
> and macOS/Linux terminals, so no WSL2 or Git Bash is required.

> **Model checkpoint:** `models/best_model.pt` is not tracked in git. **The demo
> launcher downloads it automatically** on first run (from the GitHub release, with
> checksum verification) — no manual step needed.
>
> To fetch it on its own (or if the automatic download fails):
>
> ```bash
> make fetch-model               # or: python demo/launch.py fetch-model
> # or manually:
> curl -L -o models/best_model.pt \
>   https://github.com/afonsocosta90/purr-duction-pipeline/releases/download/v1.0/best_model.pt
> ```
>
> SHA256: `270277f8ee0035b8d4ee1fcc74e4b178e2641b48a7078b3409c72c7724eadd51`
>
> Alternatively, run the full training pipeline from scratch: `make pipeline` (requires DVC data — see Development below).

---

## Demo Walkthrough

### Tab 1 — Live Prediction

1. Open **http://localhost:8501**
2. Drag any image from `demo/demo_data/` onto the uploader (or use your own)
3. The ResNet50 model classifies it in real time — prediction card + confidence gauge appear
4. Use the radio buttons to submit a **label correction** if the model was wrong
   → saved to `monitoring/feedback_log.csv` for the retraining loop

Sample images included: `cat_sample_1-3.jpg` and `not_cat_sample_1-2.jpg`

### Tab 2 — Pipeline Control (Closed-Loop Retraining)

**Step 1 — Inject Synthetic Drift**

Click **Inject Drift** — this runs `demo/simulate_drift.py` which fires batches of out-of-distribution images (random noise, colour gradients, solid fills, heavy blur) at the API. Each one is logged to `monitoring/inference_log.csv` with pixel statistics, building up a drift signal.

**Step 2 — Trigger Retraining**

Click **Retrain Model** — this runs `dvc repro --force` end-to-end inside the container:

```
ingest → validate → features → train → evaluate
```

Subprocess stdout streams live into the log box. When complete, a before/after metrics table appears showing the delta in accuracy, F1, precision, recall, and ROC-AUC. Promotion verdict: ✅ if `val_accuracy ≥ 0.94` **and** `val_f1 ≥ 0.93`.

### Tab 3 — Monitoring

Live inference stats from `monitoring/inference_log.csv` — label distribution bar chart, avg/min confidence, feedback error rate. Links to Grafana and Prometheus.

### Grafana Dashboard

Open **http://localhost:3001** (admin / catops) — the *CatOps* dashboard is pre-provisioned with:
- Total prediction counter and per-label breakdown
- Confidence gauge with red / amber / green thresholds
- Time-series: prediction volume, confidence trend, latency percentiles (p50 / p95 / p99)
- Label distribution bar chart and confidence histogram

---

## Project KPIs

*Numbers below are verified against the codebase — not aspirational targets.*

### At a glance

| Metric              | Value                          |
|---------------------|--------------------------------|
| Dataset images      | 7,390 (Oxford-IIIT Pet)        |
| Val accuracy gate   | ≥ 94 %                         |
| DVC pipeline stages | 5                              |
| API endpoints       | 5 (3 public + 2 internal)      |
| Grafana panels      | 11                             |
| Production code     | ~1,466 LOC across 13 modules   |

### Data & model

| Metric                   | Value                                            |
|--------------------------|--------------------------------------------------|
| Images                   | 7,390 — Oxford-IIIT Pet (cat vs. dog)            |
| Classes                  | 2 (`cat`, `not_cat`)                             |
| Train / val / test split | 70 / 15 / 15 (stratified, seed 42)               |
| Input size               | 224 × 224, ImageNet normalisation                |
| Architecture             | ResNet50 (`IMAGENET1K_V1`) + dropout 0.2         |
| Promotion gate           | `val_accuracy ≥ 0.94` ∧ `val_f1 ≥ 0.93`          |

### Pipeline & quality gates

| Metric                | Value                                                                              |
|-----------------------|------------------------------------------------------------------------------------|
| DVC stages            | 5 (ingest → validate → features → train → evaluate)                                |
| Validation gates      | 4 pre-split + 1 stratification leak check                                          |
| Hydra configs         | 4 (`config` / `data` / `model` / `training`)                                       |
| Determinism           | fixed seed + cached stages — `dvc repro` is a no-op when inputs are unchanged      |

### Serving

| Metric                 | Value                                                                              |
|------------------------|------------------------------------------------------------------------------------|
| FastAPI endpoints      | 5 — 3 public (`/predict`, `/health`, `/metrics`) + 2 internal                      |
| Accepted content types | 4 (`image/jpeg`, `image/png`, `image/webp`, `image/gif`)                           |
| Upload guards          | 10 MB size limit + 4 MP decompression-bomb cap                                     |
| Custom Prom metrics    | 3 (`catops_predictions_total`, `catops_prediction_confidence`, `catops_drift_score`) |
| Auto-instrumented      | HTTP latency / request counters via `prometheus-fastapi-instrumentator`            |
| Container hardening    | non-root user (uid 1001), uvicorn 2-worker                                         |

### Monitoring & drift

| Metric                | Value                                                                              |
|-----------------------|------------------------------------------------------------------------------------|
| Inference log columns | 10 (timestamp · MD5 hash · 3×RGB mean · 3×RGB std · label · confidence) — **no raw image is ever stored** (GDPR-safe) |
| Drift detector        | Evidently AI `DataDriftPreset`, scheduled daily 08:00 UTC                          |
| Drift thresholds      | drift score > 0.5 **or** rolling confidence < 0.80 (window 200, min 50 samples)    |
| Dashboard             | 11-panel Grafana *CatOps* board (label distribution · confidence histogram · p50/p95/p99 latency) |
| Alerting              | Slack webhook + uploaded `drift_report.html` artifact                              |

### CI/CD & engineering

| Metric                 | Value                                                                              |
|------------------------|------------------------------------------------------------------------------------|
| GH Actions workflows   | 2 (`ci-cd.yml` + scheduled `drift.yml`)                                            |
| CI/CD jobs             | 4 (`changes` / `quality` / `pipeline` / `docker`)                                  |
| Path-filter categories | 2 (`code` · `docker`) — `pipeline` / `docker` jobs only run on relevant changes    |
| Code quality           | ruff + black + mypy + 3 DVC pre-commit hooks                                       |
| Smoke tests            | 4 (structure · configs · `dvc.yaml` · processed splits)                            |
| Demo stack             | 4 services + 3 Streamlit tabs + 4 cross-platform launchers (`make demo`, `demo.cmd`, `demo.sh`, `python demo/launch.py`) |
| Production code        | ~1,466 LOC across 13 modules in `src/catops/`                                      |
| Repo activity          | 83 commits on `main`, 9 phases delivered                                           |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Pipeline                           │
│   data/raw/ ──► ingest.py ──► validate.py ──► build_features.py│
│   (DVC-versioned)           (quality gates)  (stratified split) │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                    Training & Evaluation                         │
│   CatDataset ──► ResNet50 ──► evaluate.py ──► MLflow registry   │
│   (split-aware)   (transfer)  (acc·F1·AUC)   (promotion gate)  │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                       Serving & Monitoring                       │
│   FastAPI /predict ──► inference_logger ──► Evidently AI drift  │
│   /health · /metrics    (pixel stats CSV)   Prometheus · Grafana│
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                        Demo Stack (Phase 9)                      │
│   Streamlit UI ──► live predict · feedback · drift inject        │
│                    dvc repro streaming · before/after metrics    │
└─────────────────────────────────────────────────────────────────┘
```

Full architecture diagrams and sequence flows: [docs/Architecture.md](docs/Architecture.md)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dependency management | Poetry + Python 3.12 |
| Data versioning | DVC (Git-like semantics for large files) |
| Config management | Hydra + OmegaConf |
| Training | PyTorch 2 · ResNet50 transfer learning |
| Experiment tracking | MLflow (params, metrics, artifacts, model registry) |
| Evaluation | scikit-learn · confusion matrix · ROC curve |
| Serving | FastAPI · Prometheus instrumentation · non-root Docker |
| CI/CD | GitHub Actions (lint → pipeline → Docker push) |
| Drift detection | Evidently AI · daily scheduled drift reports · Slack alerts |
| Observability | Prometheus + Grafana (pre-provisioned dashboard) |
| Demo UI | Streamlit + Plotly · 4-service Docker Compose stack |

---

## Project Structure

```
purr-duction-pipeline/
├── src/catops/
│   ├── data/          ingest.py · validate.py · dataset.py
│   ├── features/      build_features.py
│   ├── training/      train.py
│   ├── evaluation/    evaluate.py
│   ├── serving/       service.py · model_utils.py
│   └── monitoring/    inference_logger.py · drift.py · alerts.py
├── demo/
│   ├── streamlit_app.py          Streamlit UI (predict · pipeline · monitoring)
│   ├── simulate_drift.py         synthetic OOD image injection
│   ├── Dockerfile.demo           extends API image, adds demo deps
│   ├── docker-compose.demo.yml   4-service stack
│   └── demo_data/                sample cat + not_cat images
├── configs/                      Hydra: model.yaml · training.yaml
├── data/                         DVC-versioned raw + processed images
├── models/                       best_model.pt (DVC output)
├── monitoring/                   inference_log.csv · baseline_stats.csv
├── .github/workflows/            ci-cd.yml · drift.yml (daily 08:00 UTC)
├── dvc.yaml                      pipeline stage definitions
├── Makefile                      all developer commands
└── pyproject.toml                Poetry deps (includes demo group)
```

---

## Development

### Prerequisites

```bash
# Install dependencies (includes demo group: streamlit, httpx, plotly)
make install

# Pull DVC-versioned data
poetry run dvc pull

# Run full pipeline
make pipeline
```

### Common commands

```bash
make serve          # FastAPI on http://localhost:3000 (local, no Docker)
make api-test       # POST a sample cat image, print JSON prediction
make test           # pytest
make lint           # ruff + black check
make drift-report   # run Evidently AI drift check against inference log
make demo-logs      # follow all demo service logs
```

### Run demo UI locally without Docker

```bash
# Terminal 1
make serve

# Terminal 2
poetry install --with demo
poetry run streamlit run demo/streamlit_app.py

# Optional: inject drift manually
poetry run python demo/simulate_drift.py --count 50 --api-url http://localhost:3000
```

### Deploy demo to a remote server

```bash
DOCKER_HOST=ssh://user@your-server make demo-cloud
```

---

## Pipeline Stages

The full pipeline is defined in `dvc.yaml` and runs with `make pipeline` or `dvc repro`:

| Stage | Input | Output |
|---|---|---|
| `ingest` | `data/raw/images/` | `data/processed/cat/` + `not_cat/` |
| `validate` | processed images | `metadata.csv` · quality gates |
| `features` | `metadata.csv` | `train/val/test.csv` · `features_config.json` |
| `train` | split CSVs + configs | `models/best_model.pt` · MLflow run |
| `evaluate` | `best_model.pt` + val split | `artifacts/` (confusion matrix · ROC curve) |

Promotion gate: `val_accuracy ≥ 0.94` **and** `val_f1 ≥ 0.93` → model tagged `staging` in MLflow registry.

Data labelling convention: filename starting with an uppercase letter → `cat`, lowercase → `not_cat`. See [docs/HowToAddData.md](docs/HowToAddData.md).
