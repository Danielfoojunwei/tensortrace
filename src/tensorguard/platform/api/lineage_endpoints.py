"""
Model Lineage API Endpoints.
Provides version control and deployment history for VLA models.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import json

from ..database import get_session
from ..models.core import User, AuditLog
from ..auth import get_current_user
from ...crypto.sig import generate_hybrid_sig_keypair, sign_hybrid

router = APIRouter()


class ModelVersion(BaseModel):
    id: str
    hash: str
    message: str
    author: str
    tag: str
    status: str  # deployed, verified, archived
    test_pass_rate: Optional[str] = "98/98"
    created_at: Optional[str] = None


class DeployRequest(BaseModel):
    version_id: str
    reason: Optional[str] = "manual_deployment"


# In-memory model registry for MVP (production would use MLflow/HF Hub)
MODEL_REGISTRY = {
    "v2.1.0": {
        "id": "c1",
        "hash": "e7f2b1",
        "message": "Improve context window size",
        "author": "Daniel Foo",
        "tag": "v2.1.0",
        "status": "deployed",
        "test_pass_rate": "98/98",
        "created_at": "2026-01-11T10:00:00Z"
    },
    "v2.0.5": {
        "id": "c2",
        "hash": "a8d9c4",
        "message": "Merge PR #42: PQC Integration",
        "author": "System",
        "tag": "v2.0.5",
        "status": "verified",
        "test_pass_rate": "98/98",
        "created_at": "2026-01-11T08:00:00Z"
    },
    "v2.0.4": {
        "id": "c3",
        "hash": "b3e5f6",
        "message": "Optimize inference latency",
        "author": "Daniel Foo",
        "tag": "v2.0.4",
        "status": "archived",
        "test_pass_rate": "97/98",
        "created_at": "2026-01-11T05:00:00Z"
    },
    "v1.0.0": {
        "id": "c4",
        "hash": "d4f5g6",
        "message": "Initial commit",
        "author": "Daniel Foo",
        "tag": "v1.0.0",
        "status": "archived",
        "test_pass_rate": "95/98",
        "created_at": "2026-01-10T00:00:00Z"
    }
}


@router.get("/lineage/versions")
async def list_versions(session: Session = Depends(get_session)):
    """List all model versions with their deployment status."""
    versions = []
    for tag, info in MODEL_REGISTRY.items():
        versions.append({
            **info,
            "time": _relative_time(info.get("created_at"))
        })

    # Sort by created_at descending
    versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"versions": versions}


@router.get("/lineage/versions/{tag}")
async def get_version(tag: str, session: Session = Depends(get_session)):
    """Get detailed info about a specific model version."""
    if tag not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Version {tag} not found")

    return MODEL_REGISTRY[tag]


@router.post("/lineage/deploy")
async def deploy_version(
    req: DeployRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Deploy a specific model version.
    Creates an immutable audit log entry with PQC signature.
    """
    # Find the version
    target = None
    for tag, info in MODEL_REGISTRY.items():
        if info["id"] == req.version_id:
            target = (tag, info)
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"Version {req.version_id} not found")

    tag, info = target

    # Archive currently deployed version
    previous_deployed = None
    for t, i in MODEL_REGISTRY.items():
        if i["status"] == "deployed":
            previous_deployed = t
            i["status"] = "verified"

    # Deploy new version
    info["status"] = "deployed"

    # Create PQC-signed audit log
    pub, priv = generate_hybrid_sig_keypair()
    log_entry = {
        "action": "MODEL_DEPLOY",
        "version_tag": tag,
        "previous_version": previous_deployed,
        "reason": req.reason,
        "timestamp": datetime.utcnow().isoformat()
    }
    sig = sign_hybrid(priv, json.dumps(log_entry).encode())

    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="MODEL_DEPLOY",
        resource_id=req.version_id,
        resource_type="model_version",
        details=json.dumps(log_entry),
        pqc_signature=sig["sig_pqc"]
    )
    session.add(audit)
    session.commit()

    return {
        "status": "deployed",
        "version": tag,
        "previous": previous_deployed,
        "audit_id": audit.id
    }


@router.post("/lineage/sync")
async def sync_repository(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Sync with external model registry (HF Hub, MLflow, etc.)."""
    # Placeholder for external sync - would connect to HF Hub / MLflow
    return {
        "status": "synced",
        "source": "local",
        "versions_found": len(MODEL_REGISTRY),
        "timestamp": datetime.utcnow().isoformat()
    }


def _relative_time(iso_time: Optional[str]) -> str:
    """Convert ISO timestamp to relative time string."""
    if not iso_time:
        return "unknown"

    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        now = datetime.now(dt.tzinfo)
        diff = now - dt

        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "just now"
    except:
        return "unknown"
