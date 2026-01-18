from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select, func, case
from typing import List, Dict, Any, Optional
from ..database import get_session
from ..models.enablement_models import PolicyProfile, EnablementJob, GovernanceEvent
from ...core.privacy.ledger import PrivacyLedger
from ..services.trust_service import TrustService
from ..auth import get_current_user

router = APIRouter()
# Simple global ledger instance for the platform
# In a real distributed system, this would aggregate from agents
platform_ledger = PrivacyLedger(storage_path="./platform_privacy.json")

@router.get("/stats")
def get_stats(
    session: Session = Depends(get_session),
    current_user = Depends(get_current_user)
):
    """Get aggregate statistics for enablement dashboard."""
    # Consolidated query: count all job statuses in a single round-trip
    job_stats = session.exec(
        select(
            func.count(EnablementJob.run_id).label('total'),
            func.sum(case((EnablementJob.status == "PENDING", 1), else_=0)).label('pending'),
            func.sum(case((EnablementJob.status == "RUNNING", 1), else_=0)).label('running'),
            func.sum(case((EnablementJob.status == "SUCCESS", 1), else_=0)).label('success'),
            func.sum(case((EnablementJob.status == "FAILED", 1), else_=0)).label('failed'),
        )
    ).one()

    total_events = session.exec(select(func.count(GovernanceEvent.id))).one()

    return {
        "total_jobs": job_stats[0] or 0,
        "pending_jobs": job_stats[1] or 0,
        "running_jobs": job_stats[2] or 0,
        "success_jobs": job_stats[3] or 0,
        "failed_jobs": job_stats[4] or 0,
        "total_events": total_events,
        "privacy_consumed_epsilon": platform_ledger.total_epsilon,
        "privacy_budget_total": 10.0,  # Default budget cap
        "trust_posture": TrustService(session).get_global_posture(current_user.tenant_id)
    }

@router.get("/profiles", response_model=List[PolicyProfile])
def list_profiles(
    session: Session = Depends(get_session),
    limit: int = Query(default=100, le=1000, ge=1),
    offset: int = Query(default=0, ge=0),
):
    """List policy profiles with pagination."""
    return list(session.exec(select(PolicyProfile).offset(offset).limit(limit)).all())

@router.post("/profiles", response_model=PolicyProfile)
def create_profile(profile: PolicyProfile, session: Session = Depends(get_session)):
    session.add(profile)
    session.commit()
    session.refresh(profile)
    return profile

@router.get("/jobs", response_model=List[EnablementJob])
def list_jobs(
    session: Session = Depends(get_session),
    limit: int = Query(default=100, le=1000, ge=1),
    offset: int = Query(default=0, ge=0),
):
    """List enablement jobs with pagination."""
    return list(session.exec(select(EnablementJob).offset(offset).limit(limit)).all())

@router.get("/jobs/{run_id}", response_model=EnablementJob)
def get_job(run_id: str, session: Session = Depends(get_session)):
    job = session.get(EnablementJob, run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/events", response_model=List[GovernanceEvent])
def list_events(
    session: Session = Depends(get_session),
    limit: int = Query(default=100, le=1000, ge=1),
    offset: int = Query(default=0, ge=0),
):
    """List governance events with pagination."""
    return list(session.exec(select(GovernanceEvent).offset(offset).limit(limit)).all())
