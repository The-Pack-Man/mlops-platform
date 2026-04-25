-- MLflow uses the default mlops database, created by POSTGRES_DB env var.
-- This script sets up the feature store tables.

CREATE TABLE IF NOT EXISTS loan_features (
    id          SERIAL PRIMARY KEY,
    age                 FLOAT NOT NULL,
    income              FLOAT NOT NULL,
    loan_amount         FLOAT NOT NULL,
    credit_score        FLOAT NOT NULL,
    employment_years    FLOAT NOT NULL,
    debt_to_income      FLOAT NOT NULL,
    default_label       INTEGER NOT NULL,   -- 0 or 1
    drift_injected      BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS prediction_log (
    id              SERIAL PRIMARY KEY,
    age                 FLOAT,
    income              FLOAT,
    loan_amount         FLOAT,
    credit_score        FLOAT,
    employment_years    FLOAT,
    debt_to_income      FLOAT,
    prediction          FLOAT,
    predicted_label     INTEGER,
    model_version       TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS drift_reports (
    id              SERIAL PRIMARY KEY,
    report_json     JSONB,
    dataset_drift   BOOLEAN,
    drift_score     FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
