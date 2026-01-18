"""
FedMoE Data Models - Experts & Skills Evidence
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON, Index, UniqueConstraint
from enum import Enum
import uuid


class ExpertStatus(str, Enum):
    """Canonical expert lifecycle states."""
    ADAPTING = "adapting"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class EvidenceType(str, Enum):
    """Canonical evidence types for skill verification."""
    SIM_SUCCESS = "SIM_SUCCESS"
    REAL_WORLD_VLD = "REAL_WORLD_VLD"
    PEER_REVIEW = "PEER_REVIEW"
    VLA_BENCHMARK = "VLA_BENCHMARK"


class FedMoEExpert(SQLModel, table=True):
    """Registry of specialized FedMoE experts trained/adapted on edge nodes."""
    __table_args__ = (
        UniqueConstraint('tenant_id', 'name', 'version', name='uq_expert_tenant_name_version'),
        Index('ix_expert_tenant_status', 'tenant_id', 'status'),
        Index('ix_expert_base_model', 'base_model'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True, foreign_key="tenant.id")
    name: str = Field(index=True)  # e.g. "manipulation_grasp_v2"
    base_model: str  # e.g. "openvla-7b"
    version: str = Field(default="1.0.0")  # Semantic versioning for rollback

    status: str = Field(default=ExpertStatus.ADAPTING.value, index=True)
    accuracy_score: Optional[float] = None
    collision_rate: Optional[float] = None

    # VLA-specific metrics
    success_rate: Optional[float] = None  # Task completion rate
    avg_latency_ms: Optional[float] = None  # Inference latency
    safety_score: Optional[float] = None  # Safety validation score

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Metadata for MoE Gating
    gating_config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # Relationships
    evidences: List["SkillEvidence"] = Relationship(back_populates="expert")

class SkillEvidence(SQLModel, table=True):
    """Tamper-evident proof of skill acquisition or validation for a FedMoE expert."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    expert_id: str = Field(foreign_key="fedmoeexpert.id", index=True)
    
    evidence_type: str # SIM_SUCCESS, REAL_WORLD_VLD, PEER_REVIEW
    value_json: str = Field(default="{}")
    
    # Evidence Fabric metadata
    signed_proof: Optional[str] = None # Dilithium-3 signature of canonical evidence
    manifest_hash: Optional[str] = None
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    expert: FedMoEExpert = Relationship(back_populates="evidences")
