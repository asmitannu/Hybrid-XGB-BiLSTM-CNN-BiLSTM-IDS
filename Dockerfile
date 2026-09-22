# ── Hybrid IDS — Inference-only container ──────────────────────
# Base: Python 3.10 slim (matches the project's idsenv310 virtualenv)
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level build deps needed by some Python wheels, then clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY apps/ apps/
COPY app_models/ app_models/
COPY src/ src/

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI service via Uvicorn
CMD ["uvicorn", "apps.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
