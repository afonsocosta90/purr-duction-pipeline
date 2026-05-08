# Project Vision 

"Am I a Cat?" is a **binary image classifier** (Cat vs. Not Cat) with a complete MLOps lifecycle:

- Automated data ingestion → validation → training → promotion → serving.
- Fully reproducible, versioned, monitored, and deployed via CI/CD.
- Demonstrates every key MLOps skill recruiters look for in 2026.

## Project Pipeline

| Phase | Name                              | Core Concept                          | Status | Why It Matters |
|-------|-----------------------------------|---------------------------------------|--------|----------------|
| 1     | Project Setup & Tooling           | Reproducible environments + code organization | ✅ Done | Ensure anyone can run the project with a clear setup. |
| 2     | Data Ingestion & Versioning       | DVC + Git                             | ✅ Done | Version datasets like code. Reproducibility of experiments becomes possible. |
| 3     | Feature Engineering & Split       | Hydra configs + stratified 70/15/15 split | ✅ Done | Locked-in, leak-free data contract for all training runs. |
| 4     | Model Training & Promotion        | PyTorch ResNet50 + MLflow + automated promotion | ✅ Done | Training is automated. Only models beating accuracy + F1 thresholds are promoted. |
| 5     | Experiment Tracking               | MLflow full integration + real evaluation | ✅ Done | Real per-split metrics, artifact logging (confusion matrix, ROC curve), and promotion governed by actual val performance. |
| 6     | Pipeline Orchestration            | Prefect (DAGs)                        | Planned | Turns scripts into reliable, scheduled workflows. |
| 7     | Model Serving & API               | FastAPI + ONNX                        | Planned | Production web service with confidence scores. |
| 8     | CI/CD, Docker & Monitoring        | GitHub Actions + Evidently            | Planned | Full automation and drift detection in production. |




