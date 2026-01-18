"""
Enablement Data Models

SQLModel definitions for the Trust Enablement Platform.
Tracks Jobs, Policy configuration, and Governance events.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON

class PolicyProfile(SQLModel, table=True):
    """Configuration for governance gates."""
    id: Optional[str] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None
    
    # JSON Blob for flexible policy rules (DP limits, drift thresholds)
    rules: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class EnablementJob(SQLModel, table=True):
    """A specific execution of the Trust Pipeline."""
    run_id: str = Field(primary_key=True)
    robot_id: str = Field(index=True)
    site_id: Optional[str] = Field(default=None, index=True)
    job_type: str  # e.g., 'ingest', 'eval', 'train'
    
    status: str = Field(default="PENDING", index=True) # PENDING, RUNNING, SUCCESS, FAILED
    status_message: Optional[str] = None
    
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Metrics snapshot (privacy cost, duration, etc.)
    metrics: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    # Paths to artifacts in object storage / fs
    artifacts: Dict[str, str] = Field(default={}, sa_column=Column(JSON))

class GovernanceEvent(SQLModel, table=True):
    """Audit log of policy decisions made during jobs."""
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="enablementjob.run_id")
    
    event_type: str # 'GATE_CHECK', 'VIOLATION', 'OVERRIDE'
    gate_name: str  # 'dp_budget', 'rollback_contract'
    outcome: str    # 'PASS', 'DENY'
    details: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
