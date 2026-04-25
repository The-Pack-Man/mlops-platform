# Local MLOps Platform

A complete, self-contained model lifecycle system that runs with a single command.

```
docker compose up --build
```

No cloud account required. Everything runs on your machine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Data Layer                                                      │
│  PostgreSQL (feature store + logs) ◄── data_simulator           │
│  Redis (online feature cache)                                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  Training Layer                                                  │
│  training service ──► MLflow (tracking + registry)              │
└────────────────────┬────────────────────────────────────────────┘
                     │ champion model
┌────────────────────▼────────────────────────────────────────────┐
│  Serving Layer                                                   │
│  FastAPI /predict  /health  /metrics  /reload                   │
└────────────────────┬────────────────────────────────────────────┘
          Prometheus │ scrapes /metrics
┌────────────────────▼────────────────────────────────────────────┐
│  Monitoring Layer                                                │
│  Evidently drift reports ──► retraining trigger                 │
│  Prometheus + Grafana dashboards                                │
└─────────────────────────────────────────────────────────────────┘
```

## Service Map

| Service          | URL / Port              | Description                            |
|------------------|-------------------------|----------------------------------------|
| MLflow UI        | http://localhost:5000   | Experiment tracking + model registry   |
| Serving API      | http://localhost:8080   | FastAPI prediction endpoint            |
| Retraining API   | http://localhost:8001   | Manual + auto retraining trigger       |
| Prometheus       | http://localhost:9090   | Metrics store                          |
| Grafana          | http://localhost:3000   | Live dashboard (admin / admin)         |
| PostgreSQL       | localhost:5432          | Feature store + prediction log         |
| Redis            | localhost:6379          | Online feature cache                   |

## Quick Start

```bash
# 1. Start everything
cd mlops-platform
docker compose up --build

# 2. Wait ~3-5 minutes for:
#    - Postgres to seed 500 training rows
#    - MLflow to become healthy
#    - Training service to run and register the first model
#    - Serving service to load the champion model

# 3. Check the serving API is ready
curl http://localhost:8080/health

# 4. Make a prediction
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 65000,
    "loan_amount": 12000,
    "credit_score": 710,
    "employment_years": 8,
    "debt_to_income": 0.28
  }' | python -m json.tool
```

---

## Demo Walkthrough: Observing Drift

### Step 1 — Baseline

After startup, open:
- **Grafana** → http://localhost:3000 (admin/admin) → MLOps folder → "Loan Model Health"
- **MLflow** → http://localhost:5000 → `loan_default` experiment

Note the default rate metric (should be ~15-25% under clean data).

### Step 2 — Watch drift inject automatically

The `data_simulator` service injects covariate shift after it has produced 200 rows
post-seed. Income drops from mean ~$60k to ~$42k and credit_score from 680 to 640.

You can watch the row count:
```bash
docker compose exec postgres psql -U mlops -d mlops \
  -c "SELECT drift_injected, COUNT(*) FROM loan_features GROUP BY drift_injected;"
```

### Step 3 — Monitor drift detection

The `monitoring` service runs an Evidently drift check every 2 minutes.
Watch the logs:
```bash
docker compose logs -f monitoring
```

When drift is detected (share_of_drifted_columns ≥ 0.15), you'll see:
```
DRIFT DETECTED (score=0.33 >= threshold=0.15). Triggering retraining …
```

### Step 4 — Watch retraining

```bash
# See retraining logs
docker compose logs -f retraining

# Check retrain status via API
curl http://localhost:8001/status | python -m json.tool
```

### Step 5 — Confirm model promotion

In MLflow UI → Models → `loan_default_model`, you'll see a new version
in Production after a successful retrain that beats the champion AUC.

In Grafana, the **Current Model Version** stat panel will increment.

### Step 6 — Manual trigger

You can force a drift check + retrain at any time:
```bash
# Force retraining
curl -X POST http://localhost:8001/retrain

# Hot-reload the serving model
curl -X POST http://localhost:8080/reload
```

---

## Generate Load for Realistic Metrics

Run this in a terminal to send 200 predictions:

```bash
for i in $(seq 1 200); do
  curl -s -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d "{
      \"age\": $((RANDOM % 50 + 20)),
      \"income\": $((RANDOM % 80000 + 20000)),
      \"loan_amount\": $((RANDOM % 30000 + 5000)),
      \"credit_score\": $((RANDOM % 300 + 550)),
      \"employment_years\": $((RANDOM % 15 + 1)),
      \"debt_to_income\": 0.$((RANDOM % 60 + 10))
    }" > /dev/null
  sleep 0.1
done
echo "Done"
```

Watch the Grafana dashboard update in real time.

---

## Prometheus Metrics

The `/metrics` endpoint exposes:

| Metric | Type | Description |
|---|---|---|
| `loan_prediction_requests_total` | Counter | Total requests by status |
| `loan_prediction_latency_seconds` | Histogram | Inference latency |
| `loan_default_probability` | Histogram | Distribution of predicted probabilities |
| `loan_predicted_default_rate` | Gauge | Rolling default rate (last 100 requests) |
| `loan_model_version` | Gauge | Currently loaded model version number |

---

## Shutting Down

```bash
docker compose down          # stop containers, keep volumes
docker compose down -v       # stop + wipe all data (full reset)
```
