FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install the heavy shared deps once
RUN pip install --no-cache-dir \
    mlflow==2.13.2 \
    scikit-learn==1.5.0 \
    xgboost==2.0.3 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.131 \
    redis==5.0.4 \
    requests==2.32.3

WORKDIR /app
