FROM python:3.10-slim

# Security: Set up non-root user first
RUN groupadd -g 1000 tensorguard && \
    useradd -u 1000 -g tensorguard -s /bin/sh -m tensorguard

WORKDIR /app

# Install system dependencies (needed for crypto and ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    liboqs-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging metadata first for better layer caching
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY alembic /app/alembic
COPY alembic.ini /app/

# Install package with required extras
RUN pip install --no-cache-dir ".[fl,acme,pqc]"

# Set ownership
RUN chown -R tensorguard:tensorguard /app

# Non-root user
USER tensorguard

# Environment
ENV PYTHONPATH=/app/src
ENV PATH="/home/tensorguard/.local/bin:${PATH}"
ENV TG_ENVIRONMENT="production"
# DATABASE_URL should be provided at runtime via environment or secrets

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Default to API runner. Use 'python -m src.tensorguard.platform.worker' for worker deployments.
CMD ["uvicorn", "tensorguard.platform.main:app", "--host", "0.0.0.0", "--port", "8000"]
