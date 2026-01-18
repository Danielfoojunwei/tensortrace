"""
Enablement Sidecar Service

FastAPI wrapper for submitting and monitoring Enablement Jobs.
Deployable as a sidecar container in RobOps platforms.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import logging
import asyncio

# Import Pipeline
from tensorguard.enablement.pipelines.run_job import run_pipeline, RunContext
from tensorguard.enablement.external_platform.adapters.filesystem import FilesystemAdapter
from tensorguard.enablement.governance.policy import GovernanceEngine

app = FastAPI(title="TensorGuard Enablement Sidecar", version="1.0.0")
logger = logging.getLogger("Sidecar")

# In-memory job store (replace with DB/Redis for production)
job_store = {}

class JobSubmit(BaseModel):
    robot_id: str
    job_type: str
    config: Dict[str, Any]

class JobStatus(BaseModel):
    run_id: str
    status: str
    message: str = ""

# Adapter setup (Default to FS for sidecar, or configure via ENV)
adapter = FilesystemAdapter("./runs")
governance = GovernanceEngine({"dp_budget_limit": 100.0}) # Permissive for demo

def execute_job_bg(run_ctx: RunContext):
    """Background task wrapper."""
    try:
        run_pipeline(adapter, run_ctx, governance)
        # Status is updated by pipeline via adapter
        # But we also update local cache for API query
        # In real adapter, update_status would write to DB/File we can read.
        job_store[run_ctx.run_id] = "COMPLETED" 
    except Exception as e:
        job_store[run_ctx.run_id] = f"FAILED: {e}"

@app.post("/jobs", response_model=JobStatus)
async def submit_job(job: JobSubmit, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    
    ctx = RunContext(
        run_id=run_id,
        robot_id=job.robot_id,
        job_type=job.job_type,
        config=job.config
    )
    
    job_store[run_id] = "QUEUED"
    background_tasks.add_task(execute_job_bg, ctx)
    
    return JobStatus(run_id=run_id, status="QUEUED")

@app.get("/jobs/{run_id}", response_model=JobStatus)
async def get_job_status(run_id: str):
    if run_id not in job_store:
        # Check filesystem adapter
        try:
            p = adapter._get_run_dir(run_id) / "status.txt"
            if p.exists():
                content = p.read_text().strip()
                return JobStatus(run_id=run_id, status=content)
        except:
            pass
        raise HTTPException(status_code=404, detail="Job not found")
        
    return JobStatus(run_id=run_id, status=job_store[run_id])

@app.get("/health")
async def health():
    return {"status": "ok"}
