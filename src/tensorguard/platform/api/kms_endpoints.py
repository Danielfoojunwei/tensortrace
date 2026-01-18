"""
KMS (Key Management Service) API Endpoints.
Provides engineer control over key rotation and attestation policies.
Keys are now persisted to database instead of in-memory storage.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import secrets
import json

from ..database import get_session
from ..models.settings_models import SystemSetting, KMSKey, KMSRotationLog
from ..models.core import User, AuditLog
from ..auth import get_current_user
from ...crypto.sig import generate_hybrid_sig_keypair, sign_hybrid

router = APIRouter()


class KeyInfo(BaseModel):
    kid: str
    region: str
    created_at: str
    rotation_ttl: str
    status: str
    algorithm: str


class RotationRequest(BaseModel):
    kid: str
    reason: Optional[str] = "manual_rotation"


class KeyCreateRequest(BaseModel):
    kid: str
    region: str = "global"
    algorithm: str = "Kyber-768 + Ed25519"
    rotation_ttl_days: int = 30


# Default keys to seed the database
DEFAULT_KEYS = [
    {"kid": "key-us-east-1", "region": "us-east-1", "algorithm": "Kyber-768 + Ed25519", "rotation_ttl_days": 30},
    {"kid": "key-eu-central", "region": "eu-central-1", "algorithm": "Kyber-768 + Ed25519", "rotation_ttl_days": 30},
    {"kid": "fleet-master", "region": "global", "algorithm": "Dilithium-3", "rotation_ttl_days": 90}
]


def _ensure_default_keys(session: Session):
    """Ensure default keys exist in database."""
    for key_def in DEFAULT_KEYS:
        existing = session.get(KMSKey, key_def["kid"])
        if not existing:
            key = KMSKey(**key_def)
            session.add(key)
    session.commit()


@router.get("/kms/keys")
async def list_keys(session: Session = Depends(get_session)):
    """List all managed keys with their lifecycle status."""
    _ensure_default_keys(session)

    keys_db = session.exec(select(KMSKey)).all()
    keys = []

    for k in keys_db:
        created = k.last_rotated_at or k.created_at
        expires = created + timedelta(days=k.rotation_ttl_days)
        now = datetime.utcnow()
        days_remaining = (expires - now).days

        keys.append({
            "kid": k.kid,
            "region": k.region,
            "created_at": k.created_at.isoformat() + "Z",
            "rotation_ttl": f"{k.rotation_ttl_days}d",
            "status": k.status,
            "algorithm": k.algorithm,
            "days_remaining": max(0, days_remaining),
            "expires_at": expires.isoformat() + "Z"
        })

    return {"keys": keys}


@router.get("/kms/keys/{kid}")
async def get_key(kid: str, session: Session = Depends(get_session)):
    """Get detailed info about a specific key."""
    _ensure_default_keys(session)

    key = session.get(KMSKey, kid)
    if not key:
        raise HTTPException(status_code=404, detail=f"Key {kid} not found")

    created = key.last_rotated_at or key.created_at
    expires = created + timedelta(days=key.rotation_ttl_days)
    now = datetime.utcnow()

    # Get rotation history from logs
    logs = session.exec(
        select(KMSRotationLog)
        .where(KMSRotationLog.kid == kid)
        .order_by(KMSRotationLog.timestamp.desc())
        .limit(10)
    ).all()

    history = [
        {"timestamp": log.timestamp.isoformat(), "action": log.action, "by": log.performed_by or "system"}
        for log in logs
    ]

    # Add creation event if no history
    if not history:
        history = [{"timestamp": key.created_at.isoformat(), "action": "created", "by": "system"}]

    return {
        "kid": key.kid,
        "region": key.region,
        "created_at": key.created_at.isoformat() + "Z",
        "rotation_ttl": f"{key.rotation_ttl_days}d",
        "status": key.status,
        "algorithm": key.algorithm,
        "days_remaining": max(0, (expires - now).days),
        "expires_at": expires.isoformat() + "Z",
        "rotation_history": history
    }


@router.post("/kms/keys")
async def create_key(
    req: KeyCreateRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Create a new managed key."""
    existing = session.get(KMSKey, req.kid)
    if existing:
        raise HTTPException(status_code=400, detail=f"Key {req.kid} already exists")

    key = KMSKey(
        kid=req.kid,
        region=req.region,
        algorithm=req.algorithm,
        rotation_ttl_days=req.rotation_ttl_days,
        tenant_id=current_user.tenant_id
    )
    session.add(key)

    # Log creation
    log = KMSRotationLog(
        kid=req.kid,
        action="created",
        reason="initial_creation",
        performed_by=current_user.email
    )
    session.add(log)
    session.commit()

    return {"status": "created", "kid": req.kid}


@router.post("/kms/rotate")
async def rotate_key(
    req: RotationRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Trigger a key rotation.
    Creates an immutable audit log entry with PQC signature.
    """
    _ensure_default_keys(session)

    key = session.get(KMSKey, req.kid)
    if not key:
        raise HTTPException(status_code=404, detail=f"Key {req.kid} not found")

    old_rotated = key.last_rotated_at or key.created_at
    new_rotated = datetime.utcnow()

    # Update key
    key.last_rotated_at = new_rotated
    key.status = "active"
    session.add(key)

    # Create rotation log
    rotation_log = KMSRotationLog(
        kid=req.kid,
        action="rotated",
        reason=req.reason,
        performed_by=current_user.email
    )
    session.add(rotation_log)

    # Create PQC-signed audit log
    pub, priv = generate_hybrid_sig_keypair()
    log_entry = {
        "action": "KEY_ROTATION",
        "kid": req.kid,
        "reason": req.reason,
        "old_rotated": old_rotated.isoformat(),
        "new_rotated": new_rotated.isoformat(),
        "timestamp": datetime.utcnow().isoformat()
    }
    sig = sign_hybrid(priv, json.dumps(log_entry).encode())

    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="KEY_ROTATION",
        resource_id=req.kid,
        resource_type="kms_key",
        details=json.dumps(log_entry),
        pqc_signature=sig["sig_pqc"]
    )
    session.add(audit)

    # Sign rotation log with PQC
    rotation_log.pqc_signature = sig["sig_pqc"]
    session.add(rotation_log)

    session.commit()

    return {
        "status": "rotated",
        "kid": req.kid,
        "new_rotated_at": new_rotated.isoformat() + "Z",
        "audit_id": audit.id
    }


@router.get("/kms/attestation-policies")
async def get_attestation_policies(session: Session = Depends(get_session)):
    """Get TEE attestation policy configuration."""
    # Load from settings or use defaults
    attestation_level = "4"
    setting = session.exec(
        select(SystemSetting).where(SystemSetting.key == "attestation_level")
    ).first()
    if setting:
        attestation_level = setting.value

    return {
        "current_level": int(attestation_level),
        "levels": [
            {"level": 1, "name": "Software Only", "description": "No hardware attestation required"},
            {"level": 2, "name": "TPM 2.0", "description": "TPM-backed software claims"},
            {"level": 3, "name": "TEE Soft", "description": "Software-based enclave attestation"},
            {"level": 4, "name": "TEE Hard", "description": "Hardware-backed enclave with evidence fabric"}
        ]
    }


@router.get("/kms/rotation-schedule")
async def get_rotation_schedule(session: Session = Depends(get_session)):
    """Get the upcoming key rotation schedule."""
    _ensure_default_keys(session)

    keys = session.exec(select(KMSKey)).all()
    schedule = []

    for k in keys:
        created = k.last_rotated_at or k.created_at
        expires = created + timedelta(days=k.rotation_ttl_days)
        now = datetime.utcnow()

        schedule.append({
            "kid": k.kid,
            "algorithm": k.algorithm,
            "next_rotation": expires.isoformat() + "Z",
            "days_remaining": max(0, (expires - now).days),
            "auto_rotate": True
        })

    # Sort by days remaining
    schedule.sort(key=lambda x: x["days_remaining"])
    return {"schedule": schedule}
