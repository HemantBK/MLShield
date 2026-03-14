# MLShield Dockerfile
# Multi-stage build for minimal production image

# ---- Stage 1: Build dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies (CPU-only PyTorch to keep image small)
RUN pip install --no-cache-dir --prefix=/install \
    fastapi[standard]>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.5.0 \
    pyyaml>=6.0 \
    redis>=5.0.0 \
    torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu \
    scikit-learn>=1.3.0 \
    numpy>=1.24.0 \
    prometheus-client>=0.19.0 \
    httpx>=0.25.0 \
    structlog>=23.2.0 \
    rich>=13.7.0 \
    click>=8.1.0 \
    slowapi>=0.1.9


# ---- Stage 2: Production image ----
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY benchmark/data/models/ /app/benchmark/data/models/

# Set Python path
ENV PYTHONPATH=/app/src
ENV MLSHIELD_CONFIG=/app/configs/demo_config.yaml

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "mlshield.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
