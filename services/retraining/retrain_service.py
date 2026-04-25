"""
Retraining trigger service.
  GET  /health  — liveness
  POST /retrain — kick off a new training run (idempotent: ignores if already running)
  GET  /status  — last retraining job status
"""
import os
import time
import threading
import logging

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from fastapi import FastAPI, BackgroundTasks
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MLFLOW_URI  = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME  = os.getenv("MODEL_NAME", "loan_default_model")
SERVING_URL = os.getenv("SERVING_URL", "http://serving:8000")

PG_HOST = os.environ["POSTGRES_HOST"]
PG_PORT = int(os.environ["POSTGRES_PORT"])
PG_DB   = os.environ["POSTGRES_DB"]
PG_USER = os.environ["POSTGRES_USER"]
PG_PASS = os.environ["POSTGRES_PASSWORD"]

FEATURES = ["age", "income", "loan_amount", "credit_score",
            "employment_years", "debt_to_income"]
TARGET   = "default_label"

app = FastAPI(title="Retraining Service")

_status_lock   = threading.Lock()
_retraining    = False
_last_status   = {"status": "idle", "last_run": None, "result": None}


def get_pg():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASS
    )


def get_champion_auc(client: MlflowClient) -> float:
    try:
        prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not prod:
            return 0.0
        run = client.get_run(prod[0].run_id)
        return float(run.data.metrics.get("test_roc_auc", 0.0))
    except Exception:
        return 0.0


def promote_model(client: MlflowClient, version: str):
    for v in client.get_latest_versions(MODEL_NAME, stages=["Production"]):
        client.transition_model_version_stage(
            name=MODEL_NAME, version=v.version, stage="Archived"
        )
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage="Production"
    )
    log.info("Promoted version %s to Production.", version)


def do_retrain():
    global _retraining, _last_status

    with _status_lock:
        _retraining  = True
        _last_status = {"status": "running", "last_run": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "result": None}

    log.info("Retraining job started.")
    try:
        conn = get_pg()
        df = pd.read_sql(
            f"SELECT {', '.join(FEATURES + [TARGET])} FROM loan_features ORDER BY id",
            conn
        )
        conn.close()

        if len(df) < 100:
            raise ValueError(f"Not enough training data: {len(df)} rows")

        X = df[FEATURES].values
        y = df[TARGET].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=int(time.time()) % 10000, stratify=y
        )

        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": round(scale_pos_weight, 2),
        }
        model = xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred  = pipeline.predict(X_test)

        metrics = {
            "test_roc_auc":   roc_auc_score(y_test, y_proba),
            "test_accuracy":  accuracy_score(y_test, y_pred),
            "test_f1":        f1_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall":    recall_score(y_test, y_pred, zero_division=0),
        }

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("loan_default_retrain")

        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_param("retrain_trigger", "drift")
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )

        client = MlflowClient(MLFLOW_URI)
        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        new_ver = max(all_versions, key=lambda v: int(v.version))

        champion_auc = get_champion_auc(client)
        new_auc      = metrics["test_roc_auc"]

        promoted = False
        if new_auc > champion_auc:
            client.transition_model_version_stage(
                name=MODEL_NAME, version=new_ver.version, stage="Staging"
            )
            promote_model(client, new_ver.version)
            promoted = True
            # Tell serving to reload
            try:
                import requests as req
                req.post(f"{SERVING_URL}/reload", timeout=10)
                log.info("Serving layer notified to reload model.")
            except Exception as e:
                log.warning("Could not notify serving: %s", e)
        else:
            client.transition_model_version_stage(
                name=MODEL_NAME, version=new_ver.version, stage="Staging"
            )

        result = {
            "version": new_ver.version,
            "new_auc": round(new_auc, 4),
            "champion_auc": round(champion_auc, 4),
            "promoted": promoted,
        }
        log.info("Retrain complete: %s", result)

        with _status_lock:
            _last_status = {
                "status": "completed",
                "last_run": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "result": result,
            }

    except Exception as e:
        log.exception("Retraining failed: %s", e)
        with _status_lock:
            _last_status = {
                "status": "failed",
                "last_run": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "result": {"error": str(e)},
            }
    finally:
        with _status_lock:
            _retraining = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    with _status_lock:
        if _retraining:
            return {"status": "already_running"}
    background_tasks.add_task(do_retrain)
    return {"status": "triggered"}


@app.get("/status")
def get_status():
    with _status_lock:
        return dict(_last_status)
