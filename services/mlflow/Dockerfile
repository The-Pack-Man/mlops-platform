FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mlflow==2.13.2 psycopg2-binary boto3
EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlops:mlops@postgres:5432/mlops", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
