"""
Training pipeline — reads loan features from Postgres, trains XGBoost,
logs everything to MLflow, and promotes to Production if it beats the
current champion on held-out test data.
"""
import os
import time
import logging
import json

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import psycopg2
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
PG_HOST = os.environ["POSTGRES_HOST"]
PG_PORT = int(os.environ["POSTGRES_PORT"])
PG_DB   = os.environ["POSTGRES_DB"]
PG_USER = os.environ["POSTGRES_USER"]
PG_PASS = os.environ["POSTGRES_PASSWORD"]

MLFLOW_URI  = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME  = os.getenv("MODEL_NAME", "loan_default_model")
MIN_ROWS    = int(os.getenv("MIN_ROWS", "200"))

FEATURES = ["age", "income", "loan_amount", "credit_score",
            "employment_years", "debt_to_income"]
TARGET   = "default_label"


# ── data loading ──────────────────────────────────────────────────────────────
def load_data(use_drift_only: bool = False) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASS
    )
    where = "WHERE drift_injected = TRUE" if use_drift_only else ""
    df = pd.read_sql(
        f"SELECT {', '.join(FEATURES + [TARGET])} FROM loan_features {where} ORDER BY id",
        conn
    )
    conn.close()
    return df


# ── model building ────────────────────────────────────────────────────────────
def build_pipeline(params: dict) -> Pipeline:
    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model),
    ])


# ── MLflow helpers ────────────────────────────────────────────────────────────
def get_champion_auc(client: MlflowClient, model_name: str) -> float:
    """Return ROC-AUC of the current Production model, or 0.0 if none."""
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            return 0.0
        run_id = prod_versions[0].run_id
        run = client.get_run(run_id)
        return float(run.data.metrics.get("test_roc_auc", 0.0))
    except Exception:
        return 0.0


def promote_model(client: MlflowClient, model_name: str, version: str):
    """Archive current Production, promote new version."""
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for v in prod_versions:
        client.transition_model_version_stage(
            name=model_name, version=v.version, stage="Archived"
        )
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production"
    )
    log.info("Promoted version %s to Production.", version)


# ── main training loop ────────────────────────────────────────────────────────
def train(use_drift_data: bool = False, retrain_trigger: bool = False):
    log.info("Loading data from Postgres …")
    df = load_data(use_drift_only=False)  # always train on full history
    if len(df) < MIN_ROWS:
        log.warning("Only %d rows available (need %d). Waiting …", len(df), MIN_ROWS)
        return False

    log.info("Loaded %d rows for training.", len(df))
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
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

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("loan_default")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("retrain_trigger", retrain_trigger)

        pipeline = build_pipeline(params)
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred  = pipeline.predict(X_test)

        metrics = {
            "test_roc_auc":  roc_auc_score(y_test, y_proba),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1":       f1_score(y_test, y_pred),
            "test_precision":precision_score(y_test, y_pred, zero_division=0),
            "test_recall":   recall_score(y_test, y_pred, zero_division=0),
        }
        mlflow.log_metrics(metrics)
        log.info("Metrics: %s", metrics)

        # Log feature importance
        importances = pipeline.named_steps["clf"].feature_importances_
        importance_dict = {f: float(v) for f, v in zip(FEATURES, importances)}
        mlflow.log_dict(importance_dict, "feature_importance.json")

        # Register model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

    client = MlflowClient(MLFLOW_URI)

    # Get the version we just registered
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    new_version = max(all_versions, key=lambda v: int(v.version))

    champion_auc = get_champion_auc(client, MODEL_NAME)
    new_auc      = metrics["test_roc_auc"]

    log.info("Champion AUC=%.4f | New model AUC=%.4f", champion_auc, new_auc)

    if new_auc > champion_auc:
        # First transition to Staging
        client.transition_model_version_stage(
            name=MODEL_NAME, version=new_version.version, stage="Staging"
        )
        promote_model(client, MODEL_NAME, new_version.version)
        log.info("New champion promoted. AUC improved %.4f → %.4f", champion_auc, new_auc)
    else:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=new_version.version, stage="Staging"
        )
        log.info("New model did not beat champion; moved to Staging only.")

    return True


if __name__ == "__main__":
    # Wait for MLflow to be ready
    mlflow.set_tracking_uri(MLFLOW_URI)
    for attempt in range(30):
        try:
            mlflow.search_experiments()
            log.info("MLflow is ready.")
            break
        except Exception as e:
            log.warning("MLflow not ready (%s), retrying in 10s …", e)
            time.sleep(10)

    # Wait for enough data
    while True:
        try:
            conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                user=PG_USER, password=PG_PASS
            )
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM loan_features")
            count = cur.fetchone()[0]
            conn.close()
            if count >= MIN_ROWS:
                log.info("Found %d rows, starting training.", count)
                break
            log.info("Waiting for data: %d/%d rows …", count, MIN_ROWS)
            time.sleep(10)
        except Exception as e:
            log.warning("DB check failed: %s", e)
            time.sleep(10)

    train()
