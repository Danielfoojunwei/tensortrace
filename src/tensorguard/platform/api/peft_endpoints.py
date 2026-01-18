from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlmodel import Session, select
from typing import List, Any, Dict, Optional
from datetime import datetime
import hashlib
import json

from ..database import get_session
from ..models.core import User
from ..auth import get_current_user
from ..models.peft_models import PeftRun, PeftWizardDraft, IntegrationConfig, PeftRunStatus
from ...integrations.peft_hub.catalog import ConnectorCatalog, discover_connectors
from ...integrations.peft_hub.schemas import PeftWizardState, TrainingConfig
from ..dependencies import require_tenant_context

# Ensure connectors are registered
discover_connectors()

router = APIRouter()

@router.get("/connectors")
async def list_connectors(current_user: User = Depends(get_current_user)):
    return ConnectorCatalog.list_connectors()

@router.post("/connectors/test")
async def test_connector(connector_id: str, config: Dict[str, Any], current_user: User = Depends(get_current_user)):
    try:
        connector = ConnectorCatalog.get_connector(connector_id)
        result = connector.validate_config(config)
        return {
            "ok": result.ok,
            "details": result.details,
            "remediation": result.remediation,
            "installed": connector.check_installed()
        }
    except Exception as e:
        return {"ok": False, "details": str(e)}

@router.get("/profiles")
async def list_profiles(current_user: User = Depends(get_current_user)):
    return [
        {"id": "local-hf", "name": "Local HF Studio (No Accounts)", "description": "Uses local transformers and local filesystem storage."},
        {"id": "hf-mlflow-minio", "name": "MLOps Stack (MLflow + MinIO)", "description": "Standard enterprise stack using Docker Compose."},
        {"id": "k8s-template", "name": "Kubernetes Template Output", "description": "Generates YAML for remote training."}
    ]

@router.post("/wizard/compile")
async def compile_wizard(state: PeftWizardState, current_user: User = Depends(get_current_user)):
    # Derived defaults and validation
    config = state.dict()
    config_blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config_hash = hashlib.sha256(config_blob).hexdigest()
    config["derived_info"] = {
        "estimated_memory": "8GB (approx)",
        "config_hash": config_hash
    }
    return config

from ...integrations.peft_hub.workflow import PeftWorkflow

async def _run_workflow_task(run_id: str):
    # We need a new session in the background task
    from ..database import SessionLocal
    with SessionLocal() as session:
        workflow = PeftWorkflow(run_id, session)
        async for _ in workflow.execute():
            pass

@router.post("/runs")
async def start_run(
    state: PeftWizardState, 
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
    tenant_id: str = Depends(require_tenant_context)
):
    run = PeftRun(
        tenant_id=tenant_id,
        created_by_user_id=current_user.id,
        config_json=state.dict(),
        status=PeftRunStatus.PENDING,
        stage="INIT"
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    background_tasks.add_task(_run_workflow_task, run.id)

    return {"run_id": run.id, "status": run.status}

@router.get("/runs")
async def list_runs(
    session: Session = Depends(get_session), 
    tenant_id: str = Depends(require_tenant_context)
):
    statement = select(PeftRun).where(PeftRun.tenant_id == tenant_id).order_by(PeftRun.created_at.desc())
    return session.exec(statement).all()

@router.get("/runs/{run_id}")
async def get_run(run_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    run = session.get(PeftRun, run_id)
    if not run or run.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run

@router.post("/runs/{run_id}/promote")
async def promote_run(run_id: str, channel: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    run = session.get(PeftRun, run_id)
    if not run: raise HTTPException(status_code=404)
    if not run.metrics_json:
        return {"ok": False, "reason": "Policy Gate: Missing metrics for promotion."}
    if run.metrics_json.get("accuracy", 0) < 0.9:
        return {"ok": False, "reason": "Policy Gate: Accuracy too low for promotion."}
    
    run.stage = f"PROMOTED_{channel.upper()}"
    session.add(run)
    session.commit()
    return {"ok": True, "channel": channel}
