"""
Configuration Endpoints - Unified Agent Config Management

Endpoints for agents to fetch their configuration and for admins to manage fleet policies.
"""

import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlmodel import Session, select
from typing import Dict, Any, Optional
from packaging import version

from ..database import get_session
from ..models.core import Fleet
from ..models.settings_models import FleetPolicyRecord
from ..auth import get_current_user
from ...schemas.unified_config import AgentConfig, FleetPolicy, DeploymentDirective
from ..models.identity_models import IdentityAgent as AgentDB
from ..models.rollout_models import DeploymentPlan, DeploymentAssignment, DeploymentStatus
from .identity_endpoints import verify_fleet_auth
from ..services.trust_service import TrustService
from ..services.rollout_service import RolloutService

router = APIRouter()

# --- Admin Routes ---

@router.get("/fleets/{fleet_id}/policy", response_model=FleetPolicy)
async def get_fleet_policy(
    fleet_id: str,
    session: Session = Depends(get_session),
    current_user = Depends(get_current_user)
):
    """Get the active policy for a fleet."""
    fleet = session.get(Fleet, fleet_id)
    if not fleet or fleet.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Fleet not found")
    
    policy_record = session.exec(
        select(FleetPolicyRecord).where(
            FleetPolicyRecord.fleet_id == fleet.id,
            FleetPolicyRecord.tenant_id == current_user.tenant_id,
        )
    ).first()
    if not policy_record:
        raise HTTPException(
            status_code=404,
            detail="Fleet policy not found. Create one via the update endpoint.",
        )
    return FleetPolicy(**json.loads(policy_record.policy_json))

@router.put("/fleets/{fleet_id}/policy", response_model=FleetPolicy)
async def update_fleet_policy(
    fleet_id: str,
    policy: FleetPolicy,
    session: Session = Depends(get_session),
    current_user = Depends(get_current_user)
):
    """Update fleet policy."""
    fleet = session.get(Fleet, fleet_id)
    if not fleet or fleet.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Fleet not found")
    
    policy_record = session.exec(
        select(FleetPolicyRecord).where(
            FleetPolicyRecord.fleet_id == fleet.id,
            FleetPolicyRecord.tenant_id == current_user.tenant_id,
        )
    ).first()
    if policy_record:
        policy_record.policy_json = policy.model_dump_json()
        policy_record.updated_at = datetime.utcnow()
        policy_record.updated_by = current_user.id
    else:
        policy_record = FleetPolicyRecord(
            tenant_id=current_user.tenant_id,
            fleet_id=fleet.id,
            policy_json=policy.model_dump_json(),
            updated_by=current_user.id,
        )
    session.add(policy_record)
    session.commit()
    session.refresh(policy_record)
    return policy


# --- Agent Routes ---

# Legacy verify_agent_api_key removed in favor of verify_fleet_auth (HMAC-based)

@router.post("/agent/sync", response_model=AgentConfig)
async def sync_agent_config(
    agent_info: Dict[str, Any],
    session: Session = Depends(get_session),
    fleet: Fleet = Depends(verify_fleet_auth)
):
    """
    Agent heartbeat and config sync.

    Agent sends its local info, Server returns the authoritative config.
    Includes deployment directive if the device is part of an active deployment.
    """
    # 1. Extract agent info
    agent_name = agent_info.get("name", "unknown")
    device_id = agent_info.get("device_id")
    agent_version = agent_info.get("version", "0.0.0")

    # 2. Construct effective config using TrustService
    trust = TrustService(session).calculate_fleet_trust(fleet.id)

    # Directives based on trust score
    security_level = "high" if trust["aggregate_score"] > 85 else "medium"
    if trust["layers"]["transport"]["score"] < 50:
        # Emergency lockdown if identity is compromised or near expiry
        security_level = "fail-safe"

    # 3. Look up active deployment assignment for this device
    deployment_directive: Optional[DeploymentDirective] = None

    if device_id:
        rollout_service = RolloutService(session)

        # Find active deployment for this fleet
        active_deployment = session.exec(
            select(DeploymentPlan).where(
                DeploymentPlan.fleet_id == fleet.id,
                DeploymentPlan.status == DeploymentStatus.RUNNING.value
            )
        ).first()

        if active_deployment:
            # Get or create assignment for this device
            assignment = rollout_service.get_or_create_assignment(
                deployment=active_deployment,
                device_id=device_id
            )

            if assignment and assignment.assigned_variant != "control":
                # Get compatibility config
                compat = active_deployment.get_compatibility()
                compat_min_version = compat.get("min_agent_version", "1.0.0")

                # Compatibility check
                try:
                    if version.parse(agent_version) < version.parse(compat_min_version):
                        raise HTTPException(
                            status_code=status.HTTP_426_UPGRADE_REQUIRED,
                            detail={
                                "error": "agent_version_incompatible",
                                "required_version": compat_min_version,
                                "current_version": agent_version,
                                "deployment_id": active_deployment.id
                            }
                        )
                except version.InvalidVersion:
                    # If version parsing fails, allow but log warning
                    pass

                # Construct deployment directive
                deployment_directive = DeploymentDirective(
                    deployment_id=active_deployment.id,
                    target_adapter_id=assignment.assigned_adapter_id,
                    target_model_version=active_deployment.target_model_version,
                    shadow=assignment.is_shadow,
                    compat_min_version=compat_min_version,
                    rollback_adapter_id=active_deployment.previous_adapter_id
                )

    from ..utils.config import settings
    return AgentConfig(
        agent_name=agent_name,
        fleet_id=fleet.id,
        control_plane_url=settings.CONTROL_PLANE_URL,
        identity={
            "enabled": True,
            "scan_interval_seconds": 3600,
            "trust_score": trust["aggregate_score"]
        },
        network={
            "enabled": True,
            "defense_mode": "front" if security_level != "fail-safe" else "isolated"
        },
        ml={
            "enabled": security_level != "fail-safe",
            "model_type": "pi0",
            "security_level": security_level
        },
        deployment=deployment_directive
    )
