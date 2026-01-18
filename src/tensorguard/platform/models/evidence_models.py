"""
Community Platform Evidence & TGSP Models
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship, Session, select
from sqlalchemy import Column, JSON

class Run(SQLModel, table=True):
    run_id: str = Field(primary_key=True)
    schema_version: str = Field(default="1.0", alias="schema")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sdk_version: str
    git_commit: Optional[str] = None
    status: str = "registered"
    metrics_json: str = Field(default="{}")
    env_json: str = Field(default="{}")
    config_json: str = Field(default="{}")
    
    # Relationships
    artifacts: List["RunArtifact"] = Relationship(back_populates="run")
    policy_results: List["RunPolicyResult"] = Relationship(back_populates="run")

class RunArtifact(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.run_id")
    artifact_type: str # report.json, report.html
    sha256: str
    path: str # storage path
    
    run: Run = Relationship(back_populates="artifacts")

class PolicyPack(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    version: str
    description: Optional[str] = None
    hash: str
    is_pro: bool = False # Flag for boundary

class RunPolicyResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.run_id")
    pack_id: str = Field(foreign_key="policypack.id")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    score: float
    status: str # PASS, FAIL
    details_json: str = Field(default="{}")

    run: Run = Relationship(back_populates="policy_results")

# --- Community TGSP Models ---

class TGSPPackage(SQLModel, table=True):
    id: str = Field(primary_key=True) # Package ID from manifest
    filename: str
    producer_id: str
    created_at: datetime
    policy_id: str
    policy_version: str
    manifest_hash: str
    storage_path: str
    metadata_json: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    status: str = "uploaded" # uploaded, verified, rejected

class TGSPRelease(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    fleet_id: str = Field(index=True)
    package_id: str = Field(foreign_key="tgsppackage.id")
    channel: str = "stable" # stable, beta, alpha
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
