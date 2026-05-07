# Purr-duction Pipeline 🐱
**"Am I a Cat?"** — Production-grade MLOps pipeline for binary image classification (Cat vs Not Cat).  
A complete end-to-end demonstration of modern MLOps practices built to showcase skills that recruiters look for in 2026.

## 🎯 Current Project Status (Phase 2/8 - Data Pipeline COMPLETE)

**Achieved:**
- Poetry + DVC setup
- Fully versioned data pipeline (`ingest` → `validate`)
- **7390 high-quality images** processed and validated:
  - 2400 `cat`
  - 4990 `not_cat`
- Automated quality gates (class balance, file integrity, no duplicates, no corruption)
- Reproducible pipeline with `dvc repro`
- Git + DVC workflow
- Documentation

**Next Phase (3/8):** Feature engineering + Hydra config + stratified train/val/test split


## 🛠️ Tech Stack (Current)
| Layer              | Technology                          |
|--------------------|-------------------------------------|
| **Environment**    | Poetry + Python 3.12                |
| **Data Versioning**| DVC + Git                           |
| **Validation**     | Lightweight production quality gates|
| **Config**         | Hydra (Phase 3)                     |
| **Training**       | PyTorch (Phase 4+)                  |


## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/afonsocosta90/purr-duction-pipeline.git
cd purr-duction-pipeline

# 2. Install
make install

# 3. Enter environment
make shell

# 4. Pull latest data
poetry run dvc pull

# 5. Run full data pipeline
poetry run dvc repro --force