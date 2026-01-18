"""
Pipeline Configuration API Endpoints.
Provides engineer control over all privacy pipeline stage parameters.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import Dict, Any
from pydantic import BaseModel
from datetime import datetime

from ..database import get_session
from ..models.settings_models import SystemSetting
from ..auth import get_current_user
from ..models.core import User

router = APIRouter()

# Default pipeline configuration
DEFAULT_PIPELINE_CONFIG = {
    # GATE Stage
    "gate_threshold": "0.1",
    
    # PRIVACY Stage
    "max_norm": "1.0",
    "sparsity_ratio": "0.01",
    
    # SHIELD Stage
    "compression_ratio": "32",
    "mse_threshold": "0.05",
    
    # KMS Stage
    "rotation_ttl_days": "30",
    "attestation_level": "4",
    
    # Policy Engine
    "active_policy_pack": "soc2-evidence-pack"
}


class PipelineConfigUpdate(BaseModel):
    key: str
    value: str


@router.get("/pipeline/config")
async def get_pipeline_config(session: Session = Depends(get_session)):
    """
    Fetch the complete pipeline configuration.
    Returns defaults merged with persisted values.
    """
    config = DEFAULT_PIPELINE_CONFIG.copy()
    
    # Load persisted overrides
    settings = session.exec(
        select(SystemSetting).where(
            SystemSetting.key.in_(list(DEFAULT_PIPELINE_CONFIG.keys()))
        )
    ).all()
    
    for s in settings:
        config[s.key] = s.value
    
    return {
        "config": config,
        "stages": [
            {
                "id": "gate",
                "name": "Expert Gating (FedMoE)",
                "description": "Instruction-based relevance filtering",
                "parameters": [
                    {"key": "gate_threshold", "label": "Gate Threshold", "type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": float(config["gate_threshold"])}
                ]
            },
            {
                "id": "privacy",
                "name": "Differential Privacy",
                "description": "Gradient clipping and sparsification",
                "parameters": [
                    {"key": "max_norm", "label": "Max Gradient Norm", "type": "number", "min": 0.1, "max": 10.0, "value": float(config["max_norm"])},
                    {"key": "sparsity_ratio", "label": "Sparsity Ratio", "type": "slider", "min": 0.001, "max": 0.5, "step": 0.001, "value": float(config["sparsity_ratio"])}
                ]
            },
            {
                "id": "shield",
                "name": "Cryptographic Shield",
                "description": "Compression and quality monitoring",
                "parameters": [
                    {"key": "compression_ratio", "label": "Compression Ratio", "type": "select", "options": [8, 16, 32, 64], "value": int(config["compression_ratio"])},
                    {"key": "mse_threshold", "label": "MSE Threshold", "type": "number", "min": 0.01, "max": 0.5, "value": float(config["mse_threshold"])}
                ]
            },
            {
                "id": "kms",
                "name": "Key Management",
                "description": "TEE attestation and key rotation",
                "parameters": [
                    {"key": "rotation_ttl_days", "label": "Rotation TTL (days)", "type": "number", "min": 1, "max": 365, "value": int(config["rotation_ttl_days"])},
                    {"key": "attestation_level", "label": "Attestation Level", "type": "select", "options": [1, 2, 3, 4], "value": int(config["attestation_level"])}
                ]
            }
        ]
    }


@router.put("/pipeline/config")
async def update_pipeline_config(
    req: PipelineConfigUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Update a pipeline configuration parameter."""
    if req.key not in DEFAULT_PIPELINE_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown config key: {req.key}")
    
    setting = session.exec(
        select(SystemSetting).where(SystemSetting.key == req.key)
    ).first()
    
    if setting:
        setting.value = req.value
        setting.updated_at = datetime.utcnow()
        setting.updated_by = current_user.id
    else:
        setting = SystemSetting(
            key=req.key,
            value=req.value,
            description=f"Pipeline config: {req.key}",
            updated_by=current_user.id
        )
    
    session.add(setting)
    session.commit()
    session.refresh(setting)
    
    return {"key": setting.key, "value": setting.value, "updated_at": setting.updated_at.isoformat()}


@router.post("/pipeline/config/reset")
async def reset_pipeline_config(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Reset all pipeline configuration to defaults."""
    for key in DEFAULT_PIPELINE_CONFIG.keys():
        setting = session.exec(
            select(SystemSetting).where(SystemSetting.key == key)
        ).first()
        if setting:
            session.delete(setting)
    
    session.commit()
    return {"status": "reset", "config": DEFAULT_PIPELINE_CONFIG}
