"""
Tier 2: FedMoE Skills Library & Version Control API.
Manages expert versions, rollback, and retrieval.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import json

from ..database import get_session
from ..models.fedmoe_models import FedMoEExpert
from ..auth import get_current_user
from ..models.core import User, AuditLog
from ...crypto.sig import generate_hybrid_sig_keypair, sign_hybrid

router = APIRouter()

class ExpertVersionUpdate(BaseModel):
    expert_id: str
    target_version: str
    reason: str

@router.get("/skills/library")
async def get_skills_library_full(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get full skills library with version history.
    Groups experts by name/base_model to show version lineages.
    """
    experts = session.exec(select(FedMoEExpert).where(
        FedMoEExpert.tenant_id == current_user.tenant_id
    )).all()
    
    # Group by name (assuming name implies the "skill" family)
    library = {}
    for e in experts:
        key = f"{e.name}::{e.base_model}"
        if key not in library:
            library[key] = {
                "name": e.name,
                "base_model": e.base_model,
                "versions": [],
                "active_version": None
            }
        
        # Check if this is currently deployed/active (simplified logic)
        is_active = e.status == "deployed"
        
        library[key]["versions"].append({
            "id": e.id,
            "version": e.version,
            "status": e.status,
            "created_at": e.created_at.isoformat(),
            "accuracy": e.accuracy_score,
            "evidence_count": len(e.evidences)
        })
        
        if is_active:
            library[key]["active_version"] = e.version

    # Sort versions
    for key in library:
        library[key]["versions"].sort(key=lambda x: x["created_at"], reverse=True)
        # Default active to latest if none deployed
        if not library[key]["active_version"] and library[key]["versions"]:
            library[key]["active_version"] = library[key]["versions"][0]["version"]

    return list(library.values())

@router.post("/skills/rollback")
async def rollback_skill(
    req: ExpertVersionUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Rollback a skill to a previous version.
    This works by setting the status of the target version to 'deployed'
    and archiving the others in the same family.
    """
    target_expert = session.exec(select(FedMoEExpert).where(
        FedMoEExpert.id == req.expert_id,
        FedMoEExpert.tenant_id == current_user.tenant_id
    )).first()
    
    if not target_expert:
        raise HTTPException(status_code=404, detail="Target expert version not found")
        
    # Find all experts in this family (same name/base_model)
    family = session.exec(select(FedMoEExpert).where(
        FedMoEExpert.name == target_expert.name,
        FedMoEExpert.base_model == target_expert.base_model,
        FedMoEExpert.tenant_id == current_user.tenant_id
    )).all()
    
    previous_active = None
    for e in family:
        if e.status == "deployed":
            previous_active = e
            e.status = "archived"
            session.add(e)
            
    target_expert.status = "deployed"
    session.add(target_expert)
    
    # Audit Log for Rollback
    pub, priv = generate_hybrid_sig_keypair()
    log_entry = {
        "action": "SKILL_ROLLBACK",
        "skill_name": target_expert.name,
        "from_version": previous_active.version if previous_active else "none",
        "to_version": target_expert.version,
        "reason": req.reason,
        "timestamp": datetime.utcnow().isoformat()
    }
    sig = sign_hybrid(priv, json.dumps(log_entry).encode())
    
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="SKILL_ROLLBACK",
        resource_id=target_expert.id,
        resource_type="fedmoe_expert",
        details=json.dumps(log_entry),
        pqc_signature=sig["sig_pqc"]
    )
    session.add(audit)
    
    session.commit()
    return {"status": "success", "active_version": target_expert.version, "audit_id": audit.id}
