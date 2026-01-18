"""
Deployment Admin API Endpoints

Provides REST APIs for managing deployment plans:
- Create deployment plans (draft)
- Start/promote/pause/resume deployments
- Manual rollback
- Status with guardrail metrics snapshot

All endpoints require admin authentication (get_current_user).
All operations are tenant-scoped and audited.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

from ..database import get_session
from ..models.core import User, Fleet
from ..models.rollout_models import (
    DeploymentPlan,
    DeploymentAssignment,
    RollbackEvent,
    DeploymentStatus,
    DeploymentMode,
    RollbackTriggerType,
)
from ..services.rollout_service import RolloutService
from ..services.audit import AuditService
from ..auth import get_current_user

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateDeploymentRequest(BaseModel):
    """Request body for creating a deployment plan."""
    fleet_id: str
    target_model_version: str
    target_adapter_id: Optional[str] = None
    mode: str = DeploymentMode.CANARY.value
    stages_config: Optional[Dict[str, Any]] = None
    guardrails_config: Optional[Dict[str, Any]] = None
    compatibility_config: Optional[Dict[str, Any]] = None
    previous_adapter_id: Optional[str] = None


class DeploymentResponse(BaseModel):
    """Response model for deployment plan."""
    id: str
    fleet_id: str
    tenant_id: str
    target_model_version: str
    target_adapter_id: Optional[str]
    mode: str
    status: str
    current_stage: int
    stages: Dict[str, Any]
    guardrails: Dict[str, Any]
    compatibility: Dict[str, Any]
    created_at: str
    updated_at: str


class DeploymentStatusResponse(BaseModel):
    """Response model for deployment status with metrics."""
    deployment: DeploymentResponse
    metrics: Dict[str, Any]
    assignments_count: int
    rollback_events: List[Dict[str, Any]]


class RollbackRequest(BaseModel):
    """Request body for manual rollback."""
    reason: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def _get_deployment_or_404(
    session: Session,
    deployment_id: str,
    tenant_id: str
) -> DeploymentPlan:
    """Get deployment by ID, ensuring tenant ownership."""
    deployment = session.get(DeploymentPlan, deployment_id)
    if not deployment or deployment.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deployment not found"
        )
    return deployment


def _deployment_to_response(deployment: DeploymentPlan) -> Dict[str, Any]:
    """Convert DeploymentPlan to response dict."""
    return {
        "id": deployment.id,
        "fleet_id": deployment.fleet_id,
        "tenant_id": deployment.tenant_id,
        "target_model_version": deployment.target_model_version,
        "target_adapter_id": deployment.target_adapter_id,
        "mode": deployment.mode,
        "status": deployment.status,
        "current_stage": deployment.current_stage,
        "stages": deployment.get_stages(),
        "guardrails": deployment.get_guardrails(),
        "compatibility": deployment.get_compatibility(),
        "created_at": deployment.created_at.isoformat(),
        "updated_at": deployment.updated_at.isoformat(),
    }


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/deployments", response_model=DeploymentResponse)
async def create_deployment(
    request: CreateDeploymentRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new deployment plan (in draft status).

    The deployment will not take effect until started.
    """
    # Verify fleet ownership
    fleet = session.get(Fleet, request.fleet_id)
    if not fleet or fleet.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fleet not found"
        )

    # Validate mode
    valid_modes = [m.value for m in DeploymentMode]
    if request.mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode. Must be one of: {valid_modes}"
        )

    # Default stage configuration based on mode
    stages_config = request.stages_config or {}
    if request.mode == DeploymentMode.CANARY.value:
        stages_config.setdefault("canary_pct", 10)
        stages_config.setdefault("cohort_pct", 30)
        stages_config.setdefault("full_pct", 100)
    elif request.mode == DeploymentMode.AB.value:
        stages_config.setdefault("split_pct", 50)

    # Default guardrails
    guardrails_config = request.guardrails_config or {
        "error_rate_threshold": 0.05,
        "p99_latency_threshold_ms": 500,
        "safety_event_threshold": 3,
    }

    # Create deployment
    deployment = DeploymentPlan(
        tenant_id=current_user.tenant_id,
        fleet_id=request.fleet_id,
        target_model_version=request.target_model_version,
        target_adapter_id=request.target_adapter_id,
        mode=request.mode,
        stages_json=json.dumps(stages_config),
        guardrails_json=json.dumps(guardrails_config),
        compatibility_json=json.dumps(request.compatibility_config or {}),
        previous_adapter_id=request.previous_adapter_id,
        status=DeploymentStatus.DRAFT.value,
    )
    session.add(deployment)
    session.commit()
    session.refresh(deployment)

    # Audit log
    AuditService.log(
        session=session,
        tenant_id=current_user.tenant_id,
        action="DEPLOYMENT_CREATE",
        resource_id=deployment.id,
        resource_type="deployment",
        user_id=current_user.id,
        details={
            "fleet_id": request.fleet_id,
            "mode": request.mode,
            "target_adapter_id": request.target_adapter_id,
        },
        success=True
    )

    return _deployment_to_response(deployment)


@router.get("/deployments", response_model=List[DeploymentResponse])
async def list_deployments(
    fleet_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    List deployment plans for the tenant.

    Optionally filter by fleet_id or status.
    """
    query = select(DeploymentPlan).where(
        DeploymentPlan.tenant_id == current_user.tenant_id
    )

    if fleet_id:
        # Verify fleet ownership
        fleet = session.get(Fleet, fleet_id)
        if not fleet or fleet.tenant_id != current_user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Fleet not found"
            )
        query = query.where(DeploymentPlan.fleet_id == fleet_id)

    if status_filter:
        query = query.where(DeploymentPlan.status == status_filter)

    query = query.order_by(DeploymentPlan.created_at.desc())
    deployments = session.exec(query).all()

    return [_deployment_to_response(d) for d in deployments]


@router.get("/deployments/{deployment_id}", response_model=DeploymentStatusResponse)
async def get_deployment_status(
    deployment_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get deployment status with guardrail metrics snapshot.

    Returns current deployment state, real-time metrics, and rollback history.
    """
    deployment = _get_deployment_or_404(session, deployment_id, current_user.tenant_id)

    # Get metrics from rollout service
    rollout_service = RolloutService(session)
    metrics = rollout_service.get_guardrail_metrics(deployment)

    # Count assignments
    assignments_count = session.exec(
        select(DeploymentAssignment).where(
            DeploymentAssignment.deployment_id == deployment_id
        )
    ).all()

    # Get rollback events
    rollback_events = session.exec(
        select(RollbackEvent).where(
            RollbackEvent.deployment_id == deployment_id
        ).order_by(RollbackEvent.ts.desc())
    ).all()

    return {
        "deployment": _deployment_to_response(deployment),
        "metrics": metrics,
        "assignments_count": len(assignments_count),
        "rollback_events": [
            {
                "id": e.id,
                "trigger_type": e.trigger_type,
                "trigger_details": e.get_trigger_details(),
                "ts": e.ts.isoformat(),
            }
            for e in rollback_events
        ]
    }


@router.post("/deployments/{deployment_id}/start")
async def start_deployment(
    deployment_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Start a deployment (draft -> running).

    Only draft deployments can be started.
    """
    deployment = _get_deployment_or_404(session, deployment_id, current_user.tenant_id)

    rollout_service = RolloutService(session)
    try:
        rollout_service.start_deployment(deployment, current_user.id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    return {
        "status": "started",
        "deployment_id": deployment_id,
        "mode": deployment.mode,
        "current_stage": deployment.current_stage,
    }


@router.post("/deployments/{deployment_id}/promote")
async def promote_deployment(
    deployment_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Promote deployment to next stage (canary -> cohort -> full).

    Only running deployments can be promoted.
    """
    deployment = _get_deployment_or_404(session, deployment_id, current_user.tenant_id)

    rollout_service = RolloutService(session)
    try:
        rollout_service.promote_deployment(deployment, current_user.id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    return {
        "status": "promoted",
        "deployment_id": deployment_id,
        "deployment_status": deployment.status,
        "current_stage": deployment.current_stage,
    }


@router.post("/deployments/{deployment_id}/pause")
async def pause_deployment(
    deployment_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Pause a running deployment.

    Paused deployments stop accepting new assignments but existing
    assignments remain active.
    """
    deployment = _get_deployment_or_404(session, deployment_id, current_user.tenant_id)

    rollout_service = RolloutService(session)
    try:
        rollout_service.pause_deployment(deployment, current_user.id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    return {
        "status": "paused",
        "deployment_id": deployment_id,
    }


@router.post("/deployments/{deployment_id}/resume")
async def resume_deployment(
    deployment_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Resume a paused deployment.

    Only paused deployments can be resumed.
    """
    deployment = _get_deployment_or_404(session, deployment_id, current_user.tenant_id)

    rollout_service = RolloutService(session)
    try:
        rollout_service.resume_deployment(deployment, current_user.id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    return {
        "status": "resumed",
        "deployment_id": deployment_id,
        "current_stage": deployment.current_stage,
    }


@router.post("/deployments/{deployment_id}/rollback")
async def rollback_deployment(
    deployment_id: str,
    request: RollbackRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Manually rollback a deployment.

    Reverts all assignments to the previous adapter and marks
    deployment as rolled_back.
    """
    deployment = _get_deployment_or_404(session, deployment_id, current_user.tenant_id)

    if deployment.status not in [DeploymentStatus.RUNNING.value, DeploymentStatus.PAUSED.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot rollback deployment in status {deployment.status}"
        )

    rollout_service = RolloutService(session)
    trigger_details = {
        "initiated_by": current_user.id,
        "reason": request.reason or "Manual rollback",
    }

    rollback_event = rollout_service.trigger_rollback(
        deployment=deployment,
        trigger_type=RollbackTriggerType.MANUAL.value,
        trigger_details=trigger_details,
        user_id=current_user.id
    )

    return {
        "status": "rolled_back",
        "deployment_id": deployment_id,
        "rollback_event_id": rollback_event.id,
        "previous_adapter_id": deployment.previous_adapter_id,
    }


@router.get("/deployments/{deployment_id}/assignments")
async def get_deployment_assignments(
    deployment_id: str,
    limit: int = 100,
    offset: int = 0,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get device assignments for a deployment.

    Returns list of devices assigned to this deployment with their
    cohort bucket and variant information.
    """
    deployment = _get_deployment_or_404(session, deployment_id, current_user.tenant_id)

    assignments = session.exec(
        select(DeploymentAssignment)
        .where(DeploymentAssignment.deployment_id == deployment_id)
        .offset(offset)
        .limit(limit)
    ).all()

    return {
        "deployment_id": deployment_id,
        "total": len(assignments),
        "offset": offset,
        "limit": limit,
        "assignments": [
            {
                "id": a.id,
                "device_id": a.device_id,
                "bucket": a.bucket,
                "assigned_variant": a.assigned_variant,
                "assigned_adapter_id": a.assigned_adapter_id,
                "is_shadow": a.is_shadow,
                "created_at": a.created_at.isoformat(),
            }
            for a in assignments
        ]
    }


@router.post("/deployments/{deployment_id}/check-guardrails")
async def check_deployment_guardrails(
    deployment_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Manually check guardrails for a deployment.

    Evaluates current telemetry against guardrail thresholds and
    returns whether a rollback would be triggered.
    """
    deployment = _get_deployment_or_404(session, deployment_id, current_user.tenant_id)

    rollout_service = RolloutService(session)
    should_rollback, trigger_details = rollout_service.evaluate_rollback_triggers(deployment)
    metrics = rollout_service.get_guardrail_metrics(deployment)

    return {
        "deployment_id": deployment_id,
        "should_rollback": should_rollback,
        "trigger_details": trigger_details,
        "current_metrics": metrics,
        "guardrails": deployment.get_guardrails(),
    }
