"""
FastAPI serving layer.
  GET  /health   — liveness probe
  POST /predict  — run inference; logs input + output to Postgres
  GET  /metrics  — Prometheus scrape endpoint
  POST /reload   — hot-reload champion model from MLflow registry
"""
import os
import time
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import psycopg2
import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, RedirectResponse
from mlflow import MlflowClient
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
MLFLOW_URI   = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME   = os.getenv("MODEL_NAME", "loan_default_model")
MODEL_STAGE  = os.getenv("MODEL_STAGE", "Production")

PG_HOST = os.environ["POSTGRES_HOST"]
PG_PORT = int(os.environ["POSTGRES_PORT"])
PG_DB   = os.environ["POSTGRES_DB"]
PG_USER = os.environ["POSTGRES_USER"]
PG_PASS = os.environ["POSTGRES_PASSWORD"]

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ["REDIS_PORT"])

FEATURES = ["age", "income", "loan_amount", "credit_score",
            "employment_years", "debt_to_income"]

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "loan_prediction_requests_total",
    "Total number of prediction requests",
    ["status"]
)
REQUEST_LATENCY = Histogram(
    "loan_prediction_latency_seconds",
    "Prediction endpoint latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
PREDICTION_DIST = Histogram(
    "loan_default_probability",
    "Distribution of default probabilities returned",
    buckets=[0.1 * i for i in range(11)]
)
DEFAULT_RATE = Gauge(
    "loan_predicted_default_rate",
    "Rolling predicted default rate (last 100 requests)"
)
MODEL_VERSION_GAUGE = Gauge(
    "loan_model_version",
    "Currently loaded model version"
)

# ── state ─────────────────────────────────────────────────────────────────────
_model            = None
_model_version    = "unknown"
_model_lock       = threading.Lock()
_recent_preds     = []   # circular buffer for default rate
_pg_conn          = None
_redis_client     = None


def get_pg():
    global _pg_conn
    try:
        if _pg_conn is None or _pg_conn.closed:
            _pg_conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                user=PG_USER, password=PG_PASS
            )
        return _pg_conn
    except Exception:
        _pg_conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DB,
            user=PG_USER, password=PG_PASS
        )
        return _pg_conn


def get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis_client


def load_model(stage: str = MODEL_STAGE) -> tuple:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(MLFLOW_URI)
    model_uri = f"models:/{MODEL_NAME}/{stage}"
    log.info("Loading model from %s …", model_uri)
    # Load as native sklearn pipeline so predict_proba is available
    model = mlflow.sklearn.load_model(model_uri)
    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    version = versions[0].version if versions else "unknown"
    log.info("Loaded model version %s from stage %s", version, stage)
    return model, version


def try_load_model():
    global _model, _model_version
    for attempt in range(20):
        try:
            m, v = load_model()
            with _model_lock:
                _model = m
                _model_version = v
            try:
                MODEL_VERSION_GAUGE.set(float(v))
            except ValueError:
                pass
            log.info("Model ready (version=%s).", v)
            return
        except Exception as e:
            log.warning("Model not available yet (%s), retry %d/20 in 15s …", e, attempt + 1)
            time.sleep(15)
    log.error("Could not load model after 20 attempts. /predict will return 503.")


# ── lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model in background so the container starts fast (health check passes)
    t = threading.Thread(target=try_load_model, daemon=True)
    t.start()
    yield


app = FastAPI(title="Loan Default Prediction API", lifespan=lifespan)


# ── schemas ───────────────────────────────────────────────────────────────────
class LoanFeatures(BaseModel):
    age:              float = Field(..., ge=18,    le=80)
    income:           float = Field(..., ge=15000, le=200000)
    loan_amount:      float = Field(..., ge=1000,  le=80000)
    credit_score:     float = Field(..., ge=300,   le=850)
    employment_years: float = Field(..., ge=0,     le=40)
    debt_to_income:   float = Field(..., ge=0.01,  le=0.99)


class PredictionResponse(BaseModel):
    default_probability: float
    predicted_label:     int
    model_version:       str


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_version": _model_version}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: LoanFeatures):
    if _model is None:
        REQUEST_COUNT.labels(status="unavailable").inc()
        raise HTTPException(status_code=503, detail="Model not yet loaded.")

    start = time.perf_counter()
    try:
        import pandas as pd
        row = pd.DataFrame([{
            "age":              features.age,
            "income":           features.income,
            "loan_amount":      features.loan_amount,
            "credit_score":     features.credit_score,
            "employment_years": features.employment_years,
            "debt_to_income":   features.debt_to_income,
        }])

        with _model_lock:
            proba = float(_model.predict_proba(row)[0, 1])
            ver   = _model_version

        label = int(proba >= 0.15)
        latency = time.perf_counter() - start

        # Prometheus
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(latency)
        PREDICTION_DIST.observe(proba)

        # Rolling default rate
        _recent_preds.append(label)
        if len(_recent_preds) > 100:
            _recent_preds.pop(0)
        DEFAULT_RATE.set(sum(_recent_preds) / len(_recent_preds))

        # Log to Postgres
        try:
            conn = get_pg()
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO prediction_log
                       (age, income, loan_amount, credit_score,
                        employment_years, debt_to_income,
                        prediction, predicted_label, model_version)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (features.age, features.income, features.loan_amount,
                     features.credit_score, features.employment_years,
                     features.debt_to_income, proba, label, ver)
                )
                conn.commit()
        except Exception as e:
            log.warning("Failed to log prediction to PG: %s", e)

        return PredictionResponse(
            default_probability=proba,
            predicted_label=label,
            model_version=ver,
        )

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        log.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
def reload_model(stage: str = MODEL_STAGE):
    """Hot-reload champion model from MLflow registry."""
    global _model, _model_version
    try:
        m, v = load_model(stage)
        with _model_lock:
            _model = m
            _model_version = v
        try:
            MODEL_VERSION_GAUGE.set(float(v))
        except ValueError:
            pass
        log.info("Model hot-reloaded: version=%s", v)
        return {"status": "reloaded", "version": v}
    except Exception as e:
        log.exception("Reload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
