"""
System Settings API Endpoints.
Provides GET/PUT for global platform configuration.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import Dict, Any, List
from pydantic import BaseModel
from datetime import datetime

from ..database import get_session
from ..models.settings_models import SystemSetting
from ..auth import get_current_user
from ..models.core import User

router = APIRouter()


class SettingUpdate(BaseModel):
    key: str
    value: str


@router.get("/settings", response_model=Dict[str, str])
async def get_all_settings(session: Session = Depends(get_session)):
    """Fetch all system settings as a dictionary."""
    settings = session.exec(select(SystemSetting)).all()
    return {s.key: s.value for s in settings}


@router.get("/settings/{key}")
async def get_setting(key: str, session: Session = Depends(get_session)):
    """Fetch a single setting by key."""
    setting = session.exec(select(SystemSetting).where(SystemSetting.key == key)).first()
    if not setting:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    return {"key": setting.key, "value": setting.value}


@router.put("/settings")
async def update_setting(
    req: SettingUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Update or create a system setting."""
    setting = session.exec(select(SystemSetting).where(SystemSetting.key == req.key)).first()
    
    if setting:
        setting.value = req.value
        setting.updated_at = datetime.utcnow()
        setting.updated_by = current_user.id
    else:
        setting = SystemSetting(
            key=req.key,
            value=req.value,
            updated_by=current_user.id
        )
    
    session.add(setting)
    session.commit()
    session.refresh(setting)
    return {"key": setting.key, "value": setting.value, "updated_at": setting.updated_at.isoformat()}


@router.post("/settings/bulk")
async def update_bulk_settings(
    settings: List[SettingUpdate],
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Update multiple settings at once."""
    results = []
    for req in settings:
        setting = session.exec(select(SystemSetting).where(SystemSetting.key == req.key)).first()
        if setting:
            setting.value = req.value
            setting.updated_at = datetime.utcnow()
            setting.updated_by = current_user.id
        else:
            setting = SystemSetting(key=req.key, value=req.value, updated_by=current_user.id)
        session.add(setting)
        results.append({"key": req.key, "value": req.value})
    
    session.commit()
    return {"updated": results}
