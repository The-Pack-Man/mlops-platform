"""
Monitoring service.
Every N minutes:
  1. Pulls reference data (first 500 clean rows) and current data (last 300 rows)
     from Postgres.
  2. Runs an Evidently DataDriftPreset report.
  3. Persists the report JSON to Postgres (drift_reports table).
  4. If dataset-level drift is detected AND drift_score > DRIFT_THRESHOLD,
     POSTs to the retraining trigger endpoint.
"""
import os
import time
import json
import logging

import pandas as pd
import psycopg2
import requests
import schedule

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
PG_HOST = os.environ["POSTGRES_HOST"]
PG_PORT = int(os.environ["POSTGRES_PORT"])
PG_DB   = os.environ["POSTGRES_DB"]
PG_USER = os.environ["POSTGRES_USER"]
PG_PASS = os.environ["POSTGRES_PASSWORD"]

DRIFT_THRESHOLD        = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
RETRAIN_URL            = os.getenv("RETRAINING_TRIGGER_URL", "http://retraining:8001/retrain")
CHECK_INTERVAL_MIN     = int(os.getenv("CHECK_INTERVAL_MIN", "2"))
RETRAIN_COOLDOWN_MIN   = int(os.getenv("RETRAIN_COOLDOWN_MIN", "30"))
REPORT_DIR             = os.getenv("REPORT_DIR", "/reports")
REF_ROWS               = int(os.getenv("REF_ROWS", "500"))
CURRENT_ROWS           = int(os.getenv("CURRENT_ROWS", "300"))

_last_retrain_triggered: float = 0.0

FEATURES = ["age", "income", "loan_amount", "credit_score",
            "employment_years", "debt_to_income"]

os.makedirs(REPORT_DIR, exist_ok=True)


def get_pg():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASS
    )


def load_reference(conn) -> pd.DataFrame:
    """First REF_ROWS rows (clean, no drift) as reference."""
    return pd.read_sql(
        f"SELECT {', '.join(FEATURES)} FROM loan_features "
        f"WHERE drift_injected = FALSE ORDER BY id LIMIT {REF_ROWS}",
        conn
    )


def load_current(conn) -> pd.DataFrame:
    """Most recent CURRENT_ROWS rows."""
    return pd.read_sql(
        f"SELECT {', '.join(FEATURES)} FROM loan_features ORDER BY id DESC LIMIT {CURRENT_ROWS}",
        conn
    )


def run_drift_check():
    global _last_retrain_triggered
    log.info("Running drift check …")
    try:
        conn = get_pg()
        ref     = load_reference(conn)
        current = load_current(conn)

        if len(ref) < 50 or len(current) < 50:
            log.info("Not enough data yet (ref=%d, current=%d), skipping.", len(ref), len(current))
            conn.close()
            return

        report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
        report.run(reference_data=ref, current_data=current)
        result = report.as_dict()

        # Extract top-level drift result
        dataset_drift = False
        drift_score   = 0.0
        for metric in result.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                res = metric.get("result", {})
                dataset_drift = res.get("dataset_drift", False)
                drift_score   = res.get("share_of_drifted_columns", 0.0)
                break

        log.info("Drift check: dataset_drift=%s, drift_score=%.3f", dataset_drift, drift_score)

        # Persist report
        report_json = json.dumps(result)
        ts = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(REPORT_DIR, f"drift_report_{ts}.json")
        with open(report_path, "w") as f:
            f.write(report_json)

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO drift_reports (report_json, dataset_drift, drift_score) "
                "VALUES (%s, %s, %s)",
                (report_json, dataset_drift, drift_score)
            )
            conn.commit()

        # Trigger retraining if threshold exceeded, respecting cooldown
        if drift_score >= DRIFT_THRESHOLD:
            cooldown_secs = RETRAIN_COOLDOWN_MIN * 60
            since_last = time.time() - _last_retrain_triggered
            if since_last < cooldown_secs:
                log.info(
                    "Drift detected (score=%.3f) but cooldown active — %.0f / %d min elapsed.",
                    drift_score, since_last / 60, RETRAIN_COOLDOWN_MIN
                )
            else:
                log.warning(
                    "DRIFT DETECTED (score=%.3f >= threshold=%.3f, dataset_drift=%s). Triggering retraining …",
                    drift_score, DRIFT_THRESHOLD, dataset_drift
                )
                try:
                    resp = requests.post(RETRAIN_URL, timeout=10)
                    log.info("Retraining trigger response: %s", resp.status_code)
                    _last_retrain_triggered = time.time()
                except Exception as e:
                    log.error("Failed to trigger retraining: %s", e)
        else:
            log.info("No retraining needed (score=%.3f < threshold=%.3f)", drift_score, DRIFT_THRESHOLD)

        conn.close()

    except Exception as e:
        log.exception("Drift check failed: %s", e)


if __name__ == "__main__":
    log.info("Monitoring service starting. Check interval: %d min, drift threshold: %.2f",
             CHECK_INTERVAL_MIN, DRIFT_THRESHOLD)

    # Wait for Postgres
    while True:
        try:
            conn = get_pg()
            conn.close()
            log.info("Postgres is ready.")
            break
        except Exception as e:
            log.warning("Waiting for Postgres: %s", e)
            time.sleep(5)

    # Run once immediately, then on schedule
    run_drift_check()
    schedule.every(CHECK_INTERVAL_MIN).minutes.do(run_drift_check)

    while True:
        schedule.run_pending()
        time.sleep(30)
