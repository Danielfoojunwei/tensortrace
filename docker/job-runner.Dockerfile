FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install python deps
# (In production, copy requirements.txt first)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir rosbags mcap mcap-ros2-support uvicorn fastapi

# Copy Code
COPY . .

# Install self
RUN pip install .

ENV PYTHONUNBUFFERED=1

# Default entrypoint for Job Runner pattern
ENTRYPOINT ["python", "-m", "tensorguard.enablement.pipelines.run_job"]
