---
name: Local MLOps Platform (loan default model)
repo: The-Pack-Man/mlops-platform
status: active
one_liner: One-command model lifecycle with training, registry, serving, drift detection, gated retraining, and live dashboards.
problem: Models degrade silently after deployment. Teams need drift detection tied to safe, gated retraining.
user: Portfolio demonstration (synthetic loan data).
tags: [pilot-to-prod, observability-data, shipped-evidence]
stack: [Python, XGBoost, scikit-learn, MLflow model registry, FastAPI, Evidently, Prometheus, Grafana, PostgreSQL, Redis, Docker Compose]
run: docker compose up --build
test: none

claims:
  - id: C1
    text: Trained an XGBoost classifier and registered it in MLflow with champion/challenger promotion gated on held-out ROC-AUC.
    evidence: [services/training/train.py]
  - id: C2
    text: Served the champion model via FastAPI with hot reload and Prometheus metrics (request count, latency histogram, prediction distribution, rolling default rate, model version).
    evidence: [services/serving/main.py, prometheus/prometheus.yml]
  - id: C3
    text: Ran scheduled Evidently data-drift reports that automatically trigger retraining above a drifted-column threshold.
    evidence: [services/monitoring/monitor.py]
  - id: C4
    text: Built a retraining service that promotes a new model only if it beats the current champion, otherwise holds it in Staging.
    evidence: [services/retraining/retrain_service.py]
  - id: C5
    text: Built a data simulator that injects controlled covariate shift to demonstrate the detect-retrain-promote loop end to end.
    evidence: [services/data_simulator/simulator.py]
  - id: C6
    text: Provisioned Grafana dashboards for model health.
    evidence: [grafana/]

metrics: []

decisions:
  - Promotion is gated on beating the champion, so retraining can never silently ship a worse model.
  - Drift threshold (share of drifted columns ≥ 0.15) is explicit and configurable rather than implicit.

limitations:
  - Synthetic data. Ground-truth labels are not fed back from predictions.
  - The promotion gate uses AUC only. There is no business-cost or calibration check.
  - No automated tests.
  - Uses MLflow stage transitions, which newer MLflow versions deprecate in favor of aliases.
---

## Context
Demonstrates the post-deployment half of ML: monitoring, drift, and safe retraining.
