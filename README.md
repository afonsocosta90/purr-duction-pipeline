# Purr-duction Pipeline 🐱

**"Am I a Cat?"** — Production-grade MLOps pipeline for binary image classification (Cat vs Not Cat).

A complete end-to-end demonstration of modern MLOps best practices in 2026. This project showcases how to take a simple image classification model from notebook to a fully automated, reproducible, monitored, and deployable production system.

## 🎯 Current Project Status
**Phase 9/9 — Interactive Demo & Portfolio Showcase COMPLETE** ✅  
All phases delivered. Full closed-loop MLOps system with live UI demo.

**Achieved (all 9 phases):**
- Full project scaffolding with **Poetry**, **Git**, and **DVC**
- Modular Python package structure (`src/catops/`) with `data/ingest.py` implemented
- Automated data ingestion and validation pipeline defined in `dvc.yaml`
- **7,390 high-quality images** versioned with DVC:
  - 🐱 2,400 `cat`
  - 🚫 4,990 `not_cat`
- Robust production quality gates (file integrity, duplicates, image corruption, class balance)
- Stratified 70/15/15 train/val/test split locked via DVC
- Hydra config management (`configs/data.yaml`, `model.yaml`, `training.yaml`)
- **ResNet50 transfer learning** training stage (modern weights API, split-aware via `CatDataset`)
- **Full evaluation module** (`evaluate.py`): accuracy, F1, precision, recall, ROC-AUC, confusion matrix + ROC curve artifacts logged to MLflow
- **Real promotion logic** based on held-out val split metrics (accuracy ≥ 0.94 and F1 ≥ 0.93)
- Fully reproducible pipeline via `dvc repro`
- Pre-commit hooks, `.dvcignore`, and best-practice Git workflow
- **FastAPI serving layer** (`src/catops/serving/`): `/predict`, `/health`, `/metrics` endpoints; lifespan model loading; decompression-bomb guard; content-type validation; per-label Prometheus counters + confidence histogram
- Docker production image: non-root user, uvicorn multi-worker CMD, model path configurable via env vars
- **GitHub Actions CI/CD** (`.github/workflows/ci-cd.yml`): lint → test → DVC pull → `dvc repro` → artifact upload → Docker build/push on `workflow_dispatch`
- **Evidently AI drift detection** (`monitoring/`): per-request pixel-stat logging, daily scheduled drift reports, Prometheus gauge exposure, Slack alerts
- **Interactive Streamlit demo** (`demo/`): drag-and-drop prediction, feedback submission, one-click synthetic drift injection, live retraining with streaming output, before/after metrics comparison

## 📁 Project Structure
```bash
├── configs/                 # Hydra configurations (Phase 3+)
├── data/                    # DVC-versioned data (raw + processed)
│   ├── raw.dvc
│   └── external/
├── docker/                  # Containerization files
├── docs/                    # ProjectScope.md + mermaid-diagram.svg
├── notebooks/               # Exploratory analysis
├── pipelines/               # DVC pipeline stages
├── src/catops/              # Main production Python package
│   ├── data/ingest.py
│   ├── data/dataset.py      # CatDataset — split-aware CSV loader
│   ├── features/
│   ├── evaluation/evaluate.py  # sklearn metrics + MLflow artifacts
│   ├── serving/
│   ├── utils/
│   └── __init__.py
├── tests/                   # Unit & integration tests
├── .github/workflows/       # CI/CD pipelines
├── dvc.yaml                 # Pipeline definition
├── params.yaml              # Experiment parameters
├── Makefile                 # Developer commands
└── pyproject.toml           # Poetry configuration
`````

## 🛠️ Tech Stack (Current)

| Layer | Technology | Status |
| :---: | :---: | :---: |
| Environment | Poetry + Python 3.12 | ✅ Complete |
| Data Versioning | DVC + Git | ✅ Complete |
| Pipeline | DVC stages | ✅ Complete |
| Data Validation | Custom production quality gates | ✅ Complete |
| Code Quality | pre-commit hooks | ✅ Complete |
| Containerization | Docker | ✅ Ready |
| CI/CD | GitHub Actions | ✅ Complete |
| Config Management | Hydra | ✅ Phase 3 |
| Training | PyTorch (ResNet50) | ✅ Phase 4 |
| Evaluation | scikit-learn + matplotlib + seaborn | ✅ Phase 5 |
| Experiment Tracking | MLflow (full integration) | ✅ Phase 5 |
| Serving | FastAPI + Prometheus | ✅ Phase 7 |
| Monitoring | Evidently AI + Prometheus + Grafana | ✅ Phase 8 |
| Demo UI | Streamlit + Plotly | ✅ Phase 9 |

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/afonsocosta90/purr-duction-pipeline.git
cd purr-duction-pipeline

# 2. Install dependencies
make install

# 3. Pull the latest versioned data and run the full pipeline
poetry run dvc pull
make pipeline

# 4. Start the API server
make serve        # FastAPI on http://localhost:3000

# 5. Test the endpoint
make api-test     # POST a sample cat image, prints JSON prediction

# 6. Scrape Prometheus metrics
curl http://localhost:3000/metrics
```

## Development Commands

```bash
make install        # Install dependencies
make shell          # Enter Poetry shell
make test           # Run test suite
make lint           # Run linters & pre-commit
make pipeline       # Re-run full DVC pipeline
make serve          # Start FastAPI server locally (port 3000)
make api-test       # POST a sample image to /predict and print JSON
make drift-report   # Run Evidently AI drift detection report
```

---

## 🎬 Interactive Demo (Phase 9)

One command spins up the complete local demo stack:

```bash
make demo
```

This builds and launches four services:

| Service | URL | Credentials |
|---------|-----|-------------|
| 🔮 Streamlit UI | http://localhost:8501 | — |
| 🚀 FastAPI (Swagger docs) | http://localhost:3000/docs | — |
| 📊 Grafana dashboards | http://localhost:3001 | admin / catops |
| 🔥 Prometheus | http://localhost:9090 | — |

### Demo walkthrough

#### Step 1 — Live Prediction

1. Open **http://localhost:8501** in your browser
2. Switch to the **🔮 Live Prediction** tab
3. Drag any image from `demo/demo_data/` (or your own) onto the uploader
4. The ResNet50 model classifies it in ~100 ms — prediction card + confidence gauge appear
5. Use the radio buttons to submit a **correct label** if the prediction was wrong  
   → saved to `monitoring/feedback_log.csv` for the retraining loop

#### Step 2 — Inject Synthetic Drift

1. Switch to the **🔄 Pipeline Control** tab
2. Set the count (default: 50) and click **💉 Inject Drift**  
   → `demo/simulate_drift.py` fires random-noise, gradient, solid-colour, and blurred
   images at the `/predict` API, populating `monitoring/inference_log.csv` with
   out-of-distribution pixel statistics
3. Watch the **📊 Monitoring** tab — label distribution and avg confidence update live

#### Step 3 — Trigger Retraining

1. Still on **🔄 Pipeline Control**, click **🚀 Retrain Model**  
   → runs `dvc repro --force` (ingest → validate → features → train → evaluate)  
   → subprocess output streams live in the log box
2. When complete, the **Before vs After** metrics table shows the delta in accuracy,
   F1, precision, recall, and ROC-AUC  
3. Promotion verdict displayed: ✅ *promoted to staging* (accuracy ≥ 0.94 **and** F1 ≥ 0.93)

#### Step 4 — Grafana Dashboard

Open **http://localhost:3001** (admin / catops) — the *CatOps — Am I a Cat? Live Metrics*
dashboard is pre-provisioned with:

- Total prediction counter + per-label breakdown
- Confidence gauge (avg) with red/yellow/green thresholds
- Time-series: prediction volume, confidence trend, request latency percentiles (p50/p95/p99)
- Label distribution bar chart + confidence histogram

### Other demo commands

```bash
make demo-logs    # Follow all service logs
make demo-down    # Stop the stack
make demo-reset   # Hard reset — remove volumes + images, rebuild from scratch

# Deploy to a remote server (requires Docker + SSH)
DOCKER_HOST=ssh://user@your-server make demo-cloud
```

### Running locally without Docker

```bash
# Terminal 1 — FastAPI backend
make serve

# Terminal 2 — Streamlit UI
pip install streamlit httpx plotly
streamlit run demo/streamlit_app.py

# Terminal 3 — Inject drift manually
python demo/simulate_drift.py --count 50 --api-url http://localhost:3000
```

### Screenshot / GIF guidance

> Record a GIF with [Kap](https://getkap.co/) (macOS) or [peek](https://github.com/phw/peek) (Linux):
>
> 1. `make demo` → wait for health checks to pass
> 2. Open http://localhost:8501
> 3. Record: upload cat image → submit feedback → inject drift → trigger retrain → show metrics
> 4. Export as GIF and drop into `docs/demo.gif`
> 5. Reference in this README: `![Demo](docs/demo.gif)`

---

## 📄 CV Bullet Points

The following are polished bullet points suitable for a CV or portfolio:

- **End-to-end MLOps platform** for binary image classification (ResNet50 + PyTorch) with reproducible DVC pipelines, Hydra config management, and MLflow experiment tracking — 7 390 images, >94% accuracy on held-out test set
- **Production FastAPI serving layer** with Prometheus instrumentation, non-root Docker image, and GitHub Actions CI/CD (lint → DVC pipeline → artifact upload → GHCR push) running on every commit
- **Automated drift detection** with Evidently AI: per-request pixel-stat logging (GDPR-safe MD5 hashing), daily scheduled drift reports, Prometheus gauge updates, and Slack alerts on confidence drop
- **Interactive Streamlit demo** delivering a closed-loop retraining experience: drag-and-drop prediction, feedback submission, synthetic drift injection, live `dvc repro` streaming, and before/after metric comparison — full stack launched with a single `make demo` command
- **Full observability stack**: pre-provisioned Grafana dashboard with label distribution, confidence histogram, and p50/p95/p99 latency panels; Prometheus metrics scraping; zero manual configuration required

