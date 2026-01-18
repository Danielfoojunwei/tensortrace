import os
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlmodel import Session, select
from ..database import get_session
from ..models.evidence_models import TGSPPackage, TGSPRelease
from ...tgsp import cli, container, manifest, spec
from ...tgsp.format import read_tgsp_header

router = APIRouter()

STORAGE_DIR = "storage/community_tgsp"
os.makedirs(STORAGE_DIR, exist_ok=True)

@router.post("/upload", response_model=TGSPPackage)
async def upload_tgsp(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    session: Session = Depends(get_session)
):
    # 1. Save File (Sanitize filename to prevent path traversal)
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(STORAGE_DIR, safe_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Inspect (Open Tooling)
    try:
        data = read_tgsp_header(file_path)
        m = data["manifest"]
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid TGSP container: {e}")

    # 3. Registry Entry
    pkg = TGSPPackage(
        id=m["package_id"],
        filename=safe_filename,
        producer_id=m.get("author_id") or m.get("producer_id", "unknown"),
        created_at=datetime.fromtimestamp(m["created_at"]),
        policy_id=m.get("policy_id", "none"),
        policy_version=str(m.get("policy_version", "1")),
        manifest_hash=data["header"]["hashes"]["manifest"],
        storage_path=file_path,
        metadata_json={
            "payloads": [p.get("name", "payload") for p in m.get("content_index", [])],
            "evidence": [],
            "base_models": m.get("compat_base_model_id", [])
        },
        status="uploaded"
    )
    
    # 4. Async Verify
    async def verify_bg(pkg_id: str):
        from ..api.community_tgsp import get_session
        from ...tgsp.service import TGSPService
        
        with next(get_session()) as s:
            p = s.get(TGSPPackage, pkg_id)
            if not p: return
            
            ok, msg = TGSPService.verify_package(p.storage_path)
            p.status = "verified" if ok else "rejected"
            s.add(p)
            s.commit()

    session.add(pkg)
    session.commit()
    session.refresh(pkg)
    
    background_tasks.add_task(verify_bg, pkg.id)
    return pkg

@router.get("/packages", response_model=List[TGSPPackage])
def list_packages(session: Session = Depends(get_session)):
    return session.exec(select(TGSPPackage)).all()

@router.post("/releases", response_model=TGSPRelease)
def create_release(release: TGSPRelease, session: Session = Depends(get_session)):
    # Deactivate current active release for this fleet/channel
    current = session.exec(
        select(TGSPRelease).where(
            TGSPRelease.fleet_id == release.fleet_id,
            TGSPRelease.channel == release.channel,
            TGSPRelease.is_active == True
        )
    ).all()
    for r in current:
        r.is_active = False
        session.add(r)
    
    session.add(release)
    session.commit()
    session.refresh(release)
    return release

@router.get("/fleets/{fleet_id}/current", response_model=Optional[TGSPPackage])
def get_current_fleet_package(fleet_id: str, channel: str = "stable", session: Session = Depends(get_session)):
    release = session.exec(
        select(TGSPRelease).where(
            TGSPRelease.fleet_id == fleet_id,
            TGSPRelease.channel == channel,
            TGSPRelease.is_active == True
        )
    ).first()
    
    if not release:
        return None
        
    return session.get(TGSPPackage, release.package_id)
