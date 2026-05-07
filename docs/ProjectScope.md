# Project Vision 

"Am I a Cat?" is a **binary image classifier** (Cat vs. Not Cat) with a complete MLOps lifecycle:

- Automated data ingestion → validation → training → promotion → serving.
- Fully reproducible, versioned, monitored, and deployed via CI/CD.
- Demonstrates every key MLOps skill recruiters look for in 2026.

## Project Pipeline

| Phase | Name | Core Concept | Why It Matters |
| :---: | :---: | :---: | :--- |
| 1 | Project Setup & Tooling | Reproducible environments + code organization | Ensure anyone can run the project with a clear setup.|
| 2 | Data Ingestion & Versioning | DVC (Data Version Control) + Git | Version datasets like code. Reproducibility of experiments becomes possible. |
| 3 | Data Validation & Quality Gates | Great Expectations + Deepchecks | "Garbage in → Garbage out." Catches corrupted images, class imbalance, drift before training wastes compute. |
| 4 | Pipeline Orchestration | Prefect (DAGs) | Turns a sequence of scripts into a reliable, scheduled, retryable workflow. Handles failures gracefully. |
| 5 | Experiment Tracking & Model Governance | MLflow (or W&B) + Hydra configs | Tracks every run (hyperparams, metrics, artifacts). Possible to compare "n" experiments and reproduce the best model months later. |
| 6 | Model Training & Promotion | PyTorch + automated promotion rules | Training is automated. Only models beating accuracy + fairness thresholds get promoted to “production”. |
| 7 | Model Serving & API | FastAPI + ONNX | Turns your model into a web service that can receive images and return “Cat” / “Not Cat” with confidence. |
| 8 | CI/CD, Docker & Monitoring | GitHub Actions + Docker + Evidently | Full automation: push → test → build → deploy. Plus drift detection in production so the system stays accurate over time. |




# High-Level Architecture

