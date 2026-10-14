# Single-container deployment: FastAPI serves the API and the built React frontend from one origin.
# The dataset is downloaded at build time and verified against the md5 of creditcard.csv, and the
# model is trained during the build, so the image starts with a registered, active model.

FROM node:24-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npx vite build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FG_MLFLOW_ENABLED=false FG_STATIC_DIR=/app/frontend/dist
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY backend/requirements-deploy.txt backend/requirements-deploy.txt
RUN pip install -r backend/requirements-deploy.txt
RUN mkdir -p data/raw && \
    curl -fsSL -o data/raw/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv && \
    echo "e90efcb83d69faf99fcab8b0255024de  data/raw/creditcard.csv" | md5sum -c -
COPY params.yaml ./
COPY backend/ backend/
COPY --from=frontend /app/frontend/dist frontend/dist
RUN cd backend && python -m scripts.train
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
