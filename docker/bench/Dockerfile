# docker/bench/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y make curl build-essential

# Copy packaging metadata first for better layer caching
COPY pyproject.toml README.md Makefile /app/
COPY src /app/src
COPY alembic /app/alembic
COPY alembic.ini /app/

# Install dependencies
RUN pip install --no-cache-dir ".[bench,fl,acme,pqc]"

# Set Env
ENV PYTHONPATH=/app/src

# Default Command
CMD ["make", "bench"]
