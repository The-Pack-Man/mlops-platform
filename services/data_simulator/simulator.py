"""
Data Simulator — generates loan default data and writes it to Postgres + Redis.
After DRIFT_START_AFTER rows, it starts injecting distribution drift into
income and credit_score to simulate real-world covariate shift.
"""
import os
import time
import json
import logging
import random

import numpy as np
import psycopg2
import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
PG_HOST = os.environ["POSTGRES_HOST"]
PG_PORT = int(os.environ["POSTGRES_PORT"])
PG_DB   = os.environ["POSTGRES_DB"]
PG_USER = os.environ["POSTGRES_USER"]
PG_PASS = os.environ["POSTGRES_PASSWORD"]

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ["REDIS_PORT"])

SEED_ROWS          = int(os.getenv("SEED_ROWS", "500"))      # initial training seed
DRIFT_START_AFTER  = int(os.getenv("DRIFT_START_AFTER", "200"))  # rows before drift
INTERVAL_SECONDS   = float(os.getenv("INTERVAL_SECONDS", "2"))   # rows per second

RNG = np.random.default_rng(42)


# ── helpers ────────────────────────────────────────────────────────────────────
def get_pg():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASS
    )


def get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def generate_row(drift: bool = False, rng=RNG) -> dict:
    """
    Clean distribution (no drift):
        age              ~ N(40, 10), clipped [18, 80]
        income           ~ N(60_000, 20_000), clipped [15_000, 200_000]
        loan_amount      ~ N(15_000, 8_000), clipped [1_000, 80_000]
        credit_score     ~ N(680, 80), clipped [300, 850]
        employment_years ~ N(8, 5), clipped [0, 40]
        debt_to_income   ~ N(0.30, 0.10), clipped [0.01, 0.99]

    Drift (after threshold):
        income      shifts down by ~30 % (mean 42_000)
        credit_score shifts down by ~40 pts (mean 640)
    """
    age              = float(np.clip(rng.normal(40, 10), 18, 80))
    income_mean      = 42_000 if drift else 60_000
    income           = float(np.clip(rng.normal(income_mean, 20_000), 15_000, 200_000))
    loan_amount      = float(np.clip(rng.normal(15_000, 8_000), 1_000, 80_000))
    cs_mean          = 640 if drift else 680
    credit_score     = float(np.clip(rng.normal(cs_mean, 80), 300, 850))
    employment_years = float(np.clip(rng.normal(8, 5), 0, 40))
    debt_to_income   = float(np.clip(rng.normal(0.30, 0.10), 0.01, 0.99))

    # Logistic default probability (ground truth)
    # Credit score is the dominant factor (as in real credit risk),
    # followed by DTI, income, and loan-to-income ratio.
    log_odds = (
        -1.5
        + 0.025 * (40 - age)                        # younger → riskier
        - 0.000008 * income                          # lower income → riskier
        + 0.000015 * loan_amount                     # larger loan → riskier
        - 0.004 * (credit_score - 300)               # normalised: 300→0, 850→2.2
        + 2.0   * debt_to_income                     # high DTI strongly predicts default
        - 0.04  * employment_years                   # stable employment reduces risk
    )
    p_default = 1 / (1 + np.exp(-log_odds))
    # Under drift, bump default probability
    if drift:
        p_default = min(p_default + 0.12, 0.95)
    default_label = int(rng.random() < p_default)

    return {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "default_label": default_label,
        "drift_injected": drift,
    }


def seed_postgres(conn, n: int = SEED_ROWS):
    """Insert initial clean training rows if the table is empty."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM loan_features")
        count = cur.fetchone()[0]
        if count >= n:
            log.info("Seed data already present (%d rows), skipping.", count)
            return
        log.info("Seeding %d initial rows …", n)
        rows = [generate_row(drift=False) for _ in range(n)]
        cur.executemany(
            """INSERT INTO loan_features
               (age, income, loan_amount, credit_score,
                employment_years, debt_to_income, default_label, drift_injected)
               VALUES (%(age)s, %(income)s, %(loan_amount)s, %(credit_score)s,
                       %(employment_years)s, %(debt_to_income)s,
                       %(default_label)s, %(drift_injected)s)""",
            rows,
        )
        conn.commit()
        log.info("Seed complete.")


def push_to_redis(r: redis.Redis, row: dict):
    """Cache latest feature stats in Redis for the serving layer."""
    r.hset("latest_features", mapping={k: str(v) for k, v in row.items()})
    r.lpush("feature_stream", json.dumps(row))
    r.ltrim("feature_stream", 0, 999)  # keep last 1000


def insert_row(conn, row: dict):
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO loan_features
               (age, income, loan_amount, credit_score,
                employment_years, debt_to_income, default_label, drift_injected)
               VALUES (%(age)s, %(income)s, %(loan_amount)s, %(credit_score)s,
                       %(employment_years)s, %(debt_to_income)s,
                       %(default_label)s, %(drift_injected)s)""",
            row,
        )
        conn.commit()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Waiting for Postgres …")
    while True:
        try:
            conn = get_pg()
            break
        except Exception as e:
            log.warning("Postgres not ready: %s — retrying in 3s", e)
            time.sleep(3)

    log.info("Waiting for Redis …")
    while True:
        try:
            r = get_redis()
            r.ping()
            break
        except Exception as e:
            log.warning("Redis not ready: %s — retrying in 3s", e)
            time.sleep(3)

    seed_postgres(conn)

    log.info("Starting continuous data generation (drift after %d rows) …", DRIFT_START_AFTER)
    row_count = 0
    while True:
        try:
            drift = row_count >= DRIFT_START_AFTER
            row   = generate_row(drift=drift)
            insert_row(conn, row)
            push_to_redis(r, row)
            row_count += 1
            if row_count % 50 == 0:
                log.info("Generated %d rows (drift=%s)", row_count, drift)
            time.sleep(INTERVAL_SECONDS)
        except psycopg2.OperationalError:
            log.warning("Lost PG connection, reconnecting …")
            conn = get_pg()
        except Exception as e:
            log.error("Unexpected error: %s", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
