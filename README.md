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
make install      # Install dependencies
make shell        # Enter Poetry shell
make test         # Run test suite
make lint         # Run linters & pre-commit
make pipeline     # Re-run full DVC pipeline
make serve        # Start FastAPI server locally (port 3000)
make api-test     # POST a sample image to /predict and print JSON
```

