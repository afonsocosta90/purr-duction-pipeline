# Purr-duction Pipeline 🐱

**"Am I a Cat?"** — Production-grade MLOps pipeline for binary image classification (Cat vs Not Cat).

A complete end-to-end demonstration of modern MLOps practices built to showcase skills that recruiters look for in 2026.

---

## 🎯 Project Vision

Transform a simple image classifier into a **fully automated, reproducible, monitored, and deployable** ML system including:
- Data versioning & validation
- Pipeline orchestration
- Experiment tracking
- Automated model promotion
- REST API serving
- CI/CD + Docker
- Production drift detection

---

## 🛠️ Tech Stack

| Layer                    | Technology                          |
|-------------------------|-------------------------------------|
| **Environment**         | Poetry + Python 3.11                |
| **Data Versioning**     | DVC + Git                           |
| **Orchestration**       | Prefect                             |
| **Experiment Tracking** | MLflow                              |
| **Config Management**   | Hydra + OmegaConf                   |
| **Model Training**      | PyTorch (MPS/CPU)                   |
| **Model Serving**       | FastAPI + ONNX Runtime              |
| **Validation**          | Great Expectations + Deepchecks     |
| **CI/CD**               | GitHub Actions + Docker             |
| **Monitoring**          | Evidently AI                        |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/afonsocosta90/purr-duction-pipeline.git
cd purr-duction-pipeline

# 2. Install environment
make install

# 3. Enter shell
make shell

# 4. (Later) Pull data
poetry run dvc pull

````
---

## 📁 Project Structure

```bash
purr-duction-pipeline/
├── .github/
│   └── workflows/                  # GitHub Actions CI/CD
├── configs/                        # Hydra configuration files
├── data/
│   ├── raw/                        # Original downloaded data
│   ├── processed/                  # Cleaned & transformed data
│   └── external/                   # External sources
├── docker/                         # Dockerfile and docker-compose
├── models/                         # Trained models & MLflow
├── notebooks/                      # Exploratory notebooks
├── pipelines/                      # Prefect workflow definitions
├── src/
│   └── catops/                     # Main production Python package
│       ├── __init__.py
│       ├── data/                   # Data loading & ingestion
│       ├── features/               # Feature engineering
│       ├── models/                 # Model architecture & training
│       ├── evaluation/             # Evaluation & metrics
│       ├── serving/                # FastAPI serving logic
│       └── utils/                  # Shared utilities
├── tests/                          # Unit & integration tests
├── dvc.yaml                        # DVC pipelines
├── params.yaml                     # Hydra main config
├── pyproject.toml                  # Poetry configuration
├── Makefile                        # Common development commands
├── .pre-commit-config.yaml
└── README.md