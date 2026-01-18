"""
VLA (Vision-Language-Action) Models for Robotics Humanoids.

This module provides the data models for:
- VLA model registry with versioning
- Task type classification for robotics
- Safety score tracking per model
- PQC-signed model integrity verification
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON, Index, UniqueConstraint
from enum import Enum
import uuid


class VLATaskType(str, Enum):
    """Canonical VLA task types for robotics applications."""
    PICK_AND_PLACE = "pick_and_place"
    WELDING = "welding"
    INSPECTION = "inspection"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    ASSEMBLY = "assembly"
    SORTING = "sorting"
    PACKING = "packing"
    HANDOVER = "handover"  # Human-robot handover


class VLAModelStatus(str, Enum):
    """VLA model lifecycle states."""
    STAGED = "staged"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    RECALLED = "recalled"  # Safety-related recall


class SafetyCheckStatus(str, Enum):
    """Safety validation result states."""
    PASSED = "passed"
    FAILED = "failed"
    CONDITIONAL = "conditional"  # Passed with restrictions
    PENDING = "pending"


class VLAModel(SQLModel, table=True):
    """
    Vision-Language-Action model registry entry.

    Tracks VLA models optimized for robotics humanoids with:
    - Architecture details (vision encoder, language model, action head)
    - Performance metrics (success rate, latency, safety score)
    - Deployment tracking (fleet assignments, version control)
    - PQC signatures for model integrity
    """
    __table_args__ = (
        UniqueConstraint('tenant_id', 'name', 'version', name='uq_vla_tenant_name_version'),
        Index('ix_vla_tenant_status', 'tenant_id', 'status'),
        Index('ix_vla_task_types', 'task_types'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str = Field(index=True)
    version: str = Field(default="1.0.0")
    description: Optional[str] = None

    # Model architecture
    vision_encoder: str = Field(default="ViT-L/14")
    language_model: str = Field(default="Llama-3-8B")
    action_head: str = Field(default="Diffusion-Policy")

    # Task capabilities (JSON array of VLATaskType values)
    task_types: str = Field(default="[]")  # JSON array

    # Action space configuration
    action_dim: int = Field(default=7)  # 6-DOF + gripper
    proprioception_dim: int = Field(default=14)  # Joint angles + gripper
    action_horizon: int = Field(default=16)  # Prediction horizon

    # Performance metrics (empirical, canonical)
    success_rate: float = Field(default=0.0)
    avg_latency_ms: float = Field(default=0.0)
    safety_score: float = Field(default=0.0)

    # VLA benchmark results (1000 cycle, 5-task benchmark)
    benchmark_cycles: int = Field(default=0)
    benchmark_tasks_passed: int = Field(default=0)
    benchmark_timestamp: Optional[datetime] = None

    # Deployment tracking
    tenant_id: str = Field(foreign_key="tenant.id", index=True)
    fleet_ids: Optional[str] = None  # JSON array of fleet IDs
    status: str = Field(default=VLAModelStatus.STAGED.value, index=True)

    # Model integrity
    model_hash: str = Field(default="")  # SHA-256 of model weights
    pqc_signature: Optional[str] = None  # Dilithium-3 signature
    signed_by: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    deployed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    safety_checks: List["VLASafetyCheck"] = Relationship(back_populates="model")


class VLASafetyCheck(SQLModel, table=True):
    """
    Safety validation records for VLA deployments.

    Tracks multi-dimensional safety metrics:
    - Collision-free operation rate
    - Force/torque limit compliance
    - Emergency stop response latency
    - Workspace boundary adherence
    """
    __table_args__ = (
        Index('ix_safety_model_timestamp', 'model_id', 'timestamp'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    model_id: str = Field(foreign_key="vlamodel.id", index=True)

    # Safety dimensions (0.0 to 1.0 compliance rates)
    collision_free_rate: float = Field(default=0.0)
    force_limit_compliance: float = Field(default=0.0)
    emergency_stop_latency_ms: float = Field(default=0.0)
    workspace_boundary_adherence: float = Field(default=0.0)
    human_proximity_compliance: float = Field(default=0.0)

    # Overall status
    status: str = Field(default=SafetyCheckStatus.PENDING.value, index=True)
    overall_score: float = Field(default=0.0)

    # Test conditions
    test_environment: str = Field(default="simulation")  # simulation, staging, production
    test_scenarios: int = Field(default=0)
    passed_scenarios: int = Field(default=0)
    failed_scenarios: int = Field(default=0)

    # Details
    notes: Optional[str] = None
    failure_modes: Optional[str] = None  # JSON array of failure descriptions

    # Audit trail
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    performed_by: Optional[str] = Field(default=None, foreign_key="user.id")
    pqc_signature: Optional[str] = None  # Tamper-evident signature

    # Relationship
    model: VLAModel = Relationship(back_populates="safety_checks")


class VLADeploymentLog(SQLModel, table=True):
    """
    Immutable log of VLA model deployment events.

    Tracks all deployment actions for audit and rollback purposes.
    """
    __table_args__ = (
        Index('ix_deploy_model_timestamp', 'model_id', 'timestamp'),
        Index('ix_deploy_fleet_timestamp', 'fleet_id', 'timestamp'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    model_id: str = Field(foreign_key="vlamodel.id", index=True)
    fleet_id: str = Field(index=True)

    # Deployment details
    action: str  # deployed, rolled_back, recalled
    previous_version: Optional[str] = None
    new_version: str

    # Safety gate check
    safety_check_id: Optional[str] = Field(default=None, foreign_key="vlasafetycheck.id")
    safety_approved: bool = Field(default=False)

    # Deployment metrics
    rollout_percentage: float = Field(default=100.0)  # For canary deployments
    affected_robots: int = Field(default=0)

    # Audit
    reason: Optional[str] = None
    performed_by: Optional[str] = Field(default=None, foreign_key="user.id")
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    pqc_signature: Optional[str] = None


class VLABenchmarkResult(SQLModel, table=True):
    """
    VLA benchmark results for canonical performance tracking.

    Implements the 1000-cycle, 5-task benchmark standard for
    VLA PEFT in robotics humanoids.
    """
    __table_args__ = (
        Index('ix_benchmark_model_timestamp', 'model_id', 'timestamp'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    model_id: str = Field(foreign_key="vlamodel.id", index=True)

    # Benchmark configuration
    total_cycles: int = Field(default=1000)
    tasks: str = Field(default="[]")  # JSON array of task types

    # Per-task results (JSON: {task_type: {success_rate, avg_time_s, failures}})
    task_results: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # Aggregate metrics
    overall_success_rate: float = Field(default=0.0)
    avg_cycle_time_s: float = Field(default=0.0)
    total_successes: int = Field(default=0)
    total_failures: int = Field(default=0)

    # Environment
    test_environment: str = Field(default="simulation")
    robot_type: str = Field(default="humanoid")

    # Canonical evidence
    evidence_hash: Optional[str] = None  # Hash of benchmark data
    pqc_signature: Optional[str] = None

    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    performed_by: Optional[str] = Field(default=None, foreign_key="user.id")
