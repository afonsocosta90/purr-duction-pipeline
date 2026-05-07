# Purr-duction Pipeline 🐱

**"Am I a Cat?"** — Production-grade MLOps pipeline for binary image classification (Cat vs Not Cat).

A complete end-to-end demonstration of modern MLOps best practices in 2026. This project showcases how to take a simple image classification model from notebook to a fully automated, reproducible, monitored, and deployable production system.

## 🎯 Current Project Status
**Phase 3/8 - Feature Engineering + Hydra Config + Stratified Split IN PROGRESS** ✅

**Achieved:**
- Full project scaffolding with **Poetry**, **Git**, and **DVC**
- Modular Python package structure (`src/catops/`) with `data/ingest.py` implemented
- Automated data ingestion and validation pipeline defined in `dvc.yaml`
- **7,390 high-quality images** versioned with DVC:
  - 🐱 2,400 `cat`
  - 🚫 4,990 `not_cat`
- Robust production quality gates (file integrity, duplicates, image corruption, class balance)
- Fully reproducible pipeline via `dvc repro`
- Pre-commit hooks, `.dvcignore`, and best-practice Git workflow
- Docker-ready environment (`docker/`, `.dockerignore`)
- GitHub Actions workflow foundation (`.github/workflows/`)
- Scaffolding for all remaining phases (`configs/`, `pipelines/`, `src/catops/{features,evaluation,serving,utils}/`, tests, notebooks)
- Comprehensive documentation (`docs/ProjectScope.md` + Mermaid architecture diagram)

**Next Phase (4/8):** PyTorch model training + experiment tracking + MLflow

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
│   ├── features/
│   ├── evaluation/
│   ├── serving/
│   ├── utils/
│   └── __init__.py
├── tests/                   # Unit & integration tests
├── .github/workflows/       # CI/CD pipelines
├── dvc.yaml                 # Pipeline definition
├── params.yaml              # Experiment parameters
├── Makefile                 # Developer commands
├── pyproject.toml           # Poetry configuration
└── project-tree.txt         # Current tree snapshot
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
| Training | PyTorch | ⏳ Phase 4 |
| Experiment Tracking | MLflow / Weights & Biases | ⏳ Phase 4 |
| Serving | FastAPI + ONNX / TorchServe / BentoML | ⏳ Phase 6 |

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

