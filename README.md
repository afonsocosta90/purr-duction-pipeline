# Purr-duction Pipeline 🐱

**"Am I a Cat?"** — Production-grade MLOps pipeline for binary image classification (Cat vs Not Cat).

A complete end-to-end demonstration of modern MLOps best practices in 2026. This project showcases how to take a simple image classification model from notebook to a fully automated, reproducible, monitored, and deployable production system.

## 🎯 Current Project Status
**Phase 5/8 - Experiment Tracking (MLflow full integration) COMPLETE** ✅  
**Next: Phase 6/8 - CI/CD Pipeline (GitHub Actions)**

**Achieved:**
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
- Docker-ready environment (`docker/`, `.dockerignore`)
- GitHub Actions workflow foundation (`.github/workflows/`)

**Next Phase (6/8):** CI/CD pipeline (GitHub Actions)

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
| CI/CD | GitHub Actions | ✅ Skeleton |
| Config Management | Hydra | ✅ Phase 3 |
| Training | PyTorch (ResNet50) | ✅ Phase 4 |
| Evaluation | scikit-learn + matplotlib + seaborn | ✅ Phase 5 |
| Experiment Tracking | MLflow (full integration) | ✅ Phase 5 |
| Serving | FastAPI + ONNX / TorchServe / BentoML | ⏳ Phase 7 |

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/afonsocosta90/purr-duction-pipeline.git
cd purr-duction-pipeline

# 2. Install dependencies
make install

# 3. Enter the Poetry shell
make shell

# 4. Pull the latest versioned data
poetry run dvc pull

# 5. Run the full data pipeline
poetry run dvc repro --force

# 6. Explore the processed data
ls -l data/raw/
`````

## Development Commands

```bash
make install      # Install dependencies
make shell        # Enter Poetry shell
make test         # Run test suite
make lint         # Run linters & pre-commit
make dvc-repro    # Re-run full DVC pipeline
````

