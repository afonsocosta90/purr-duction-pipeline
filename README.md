# Am I a Cat? — MLOps Portfolio Project

**Binary image classification (Cat vs Not Cat) built as a complete, production-grade MLOps platform.**

ResNet50 transfer learning · DVC pipelines · MLflow experiment tracking · FastAPI serving · Evidently AI drift detection · Prometheus + Grafana observability · GitHub Actions CI/CD · Interactive Streamlit demo

> 7,390 training images · >94% val accuracy · 9 phases · all complete ✅

---

## Live Demo — One Command

```bash
make demo
```

Builds and starts four Docker services:

| Service | URL | Notes |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Main demo — start here |
| FastAPI | http://localhost:3000/docs | Swagger / REST API |
| Grafana | http://localhost:3001 | Login: `admin` / `catops` |
| Prometheus | http://localhost:9090 | Raw metrics |

```bash
make demo-down    # stop all services (also clears session logs)
make demo-reset   # hard reset — remove volumes + images, rebuild from scratch
```

### Requirements

- Docker Desktop running
- ~4 GB disk space (PyTorch base image)
- Ports 3000, 3001, 8501, 9090 free
- **Windows**: `make` is not available natively — run commands from WSL2 or Git Bash

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

---

## CV Bullet Points

- **End-to-end MLOps platform** for binary image classification (ResNet50 + PyTorch) with reproducible DVC pipelines, Hydra config management, and MLflow experiment tracking — 7,390 images, >94% accuracy on held-out validation set
- **Production FastAPI serving layer** with Prometheus instrumentation, non-root Docker image, and GitHub Actions CI/CD (lint → DVC pipeline → artifact upload → GHCR push) running on every commit
- **Automated drift detection** with Evidently AI: per-request pixel-stat logging (GDPR-safe MD5 hashing), daily scheduled drift reports, Prometheus gauge updates, and Slack alerts on confidence drop
- **Interactive Streamlit demo** delivering a closed-loop retraining experience: drag-and-drop prediction, human feedback submission, synthetic drift injection, live `dvc repro` log streaming, and before/after metric comparison — full 4-service stack launched with a single `make demo`
- **Full observability stack**: pre-provisioned Grafana dashboard with label distribution, confidence histogram, and p50/p95/p99 latency panels; zero manual configuration required
