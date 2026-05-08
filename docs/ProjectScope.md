# Project Vision 

"Am I a Cat?" is a **binary image classifier** (Cat vs. Not Cat) with a complete MLOps lifecycle:

- Automated data ingestion → validation → training → promotion → serving.
- Fully reproducible, versioned, monitored, and deployed via CI/CD.
- Demonstrates every key MLOps skill recruiters look for in 2026.

## Project Pipeline

| Phase | Name                              | Core Concept                          | Why It Matters |
|-------|-----------------------------------|---------------------------------------|----------------|
| 1     | Project Setup & Tooling           | Reproducible environments + code organization | Ensure anyone can run the project with a clear setup. |
| 2     | Data Ingestion & Versioning       | DVC + Git                             | Version datasets like code. Reproducibility of experiments becomes possible. |
| 3     | Feature Engineering & Split       | Hydra configs + stratified 70/15/15 split | Locked-in, leak-free data contract for all training runs. |
| 4     | Model Training & Promotion        | PyTorch + automated promotion rules   | Training is automated. Only models beating accuracy + fairness thresholds get promoted. |
| 5     | Experiment Tracking               | MLflow + Hydra                        | Tracks every run and enables model governance. |
| 6     | Pipeline Orchestration            | Prefect (DAGs)                        | Turns scripts into reliable, scheduled workflows. |
| 7     | Model Serving & API               | FastAPI + ONNX                        | Production web service with confidence scores. |
| 8     | CI/CD, Docker & Monitoring        | GitHub Actions + Evidently            | Full automation and drift detection in production. |




