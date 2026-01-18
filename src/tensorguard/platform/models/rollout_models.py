"""
Rollout Models for Production Deployment Strategy

Supports:
- Canary / A-B / Shadow / Full deployment modes
- Deterministic cohort assignment (hash-based)
- Compatibility enforcement (version checks)
- Automated rollback triggers
- Staged rollout progression

All assignments use deterministic hashing for reproducibility.
"""

from typing import Optional, List
from sqlmodel import SQLModel, Field
from sqlalchemy import Index, UniqueConstraint
from datetime import datetime
from enum import Enum
import uuid
import json


# =============================================================================
# Enums
# =============================================================================

class DeploymentMode(str, Enum):
    """Deployment rollout modes."""
    CANARY = "canary"       # Small percentage first
    AB = "ab"               # A/B split testing
    SHADOW = "shadow"       # Run without acting
    FULL = "full"           # Full fleet deployment


class DeploymentStatus(str, Enum):
    """Deployment plan status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    ROLLED_BACK = "rolled_back"
    COMPLETED = "completed"


class RollbackTriggerType(str, Enum):
    """Types of rollback triggers."""
    TELEMETRY_THRESHOLD = "telemetry_threshold"
    MANUAL = "manual"
    COMPAT_VIOLATION = "compat_violation"
    SAFETY_EVENT = "safety_event"


class AssignmentVariant(str, Enum):
    """A/B test variants."""
    A = "A"  # Control
    B = "B"  # Treatment


# =============================================================================
# DeploymentPlan - Rollout Plan Definition
# =============================================================================

class DeploymentPlan(SQLModel, table=True):
    """
    Defines a deployment rollout plan for a fleet.

    Supports canary, A/B, shadow, and full deployment modes with
    configurable stages, guardrails, and compatibility requirements.
    """
    __tablename__ = "deployment_plan"
    __table_args__ = (
        Index('ix_deployment_plan_fleet_status', 'fleet_id', 'status'),
        Index('ix_deployment_plan_tenant_created', 'tenant_id', 'created_at'),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    tenant_id: str = Field(
        foreign_key="tenant.id",
        index=True
    )
    fleet_id: str = Field(
        foreign_key="fleet.id",
        index=True
    )

    # Target deployment
    target_model_version: str = Field(
        description="Target model version to deploy"
    )
    target_adapter_id: Optional[str] = Field(
        default=None,
        description="Target LoRA/PEFT adapter ID"
    )

    # Deployment mode
    mode: str = Field(
        default=DeploymentMode.CANARY.value,
        description="Deployment mode (canary/ab/shadow/full)"
    )

    # Stage configuration (JSON)
    # Example: {"canary_pct": 10, "cohort_pct": 30, "full_pct": 100}
    stages_json: str = Field(
        default="{}",
        description="Stage percentages and configuration"
    )

    # Guardrails configuration (JSON)
    # Example: {"error_rate_threshold": 0.05, "p99_latency_threshold_ms": 500}
    guardrails_json: str = Field(
        default="{}",
        description="Guardrails for automatic rollback"
    )

    # Compatibility requirements (JSON)
    # Example: {"min_agent_version": "1.2.0", "min_runtime_version": "2.0.0"}
    compatibility_json: str = Field(
        default="{}",
        description="Version and compatibility requirements"
    )

    # Status
    status: str = Field(
        default=DeploymentStatus.DRAFT.value,
        index=True,
        description="Current deployment status"
    )

    # Current stage (0=canary, 1=cohort, 2=full)
    current_stage: int = Field(
        default=0,
        description="Current rollout stage index"
    )

    # Previous adapter for rollback
    previous_adapter_id: Optional[str] = Field(
        default=None,
        description="Previous adapter ID for rollback"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow
    )

    def get_stages(self) -> dict:
        """Parse stages configuration."""
        try:
            return json.loads(self.stages_json)
        except json.JSONDecodeError:
            return {"canary_pct": 10, "cohort_pct": 30, "full_pct": 100}

    def get_guardrails(self) -> dict:
        """Parse guardrails configuration."""
        try:
            return json.loads(self.guardrails_json)
        except json.JSONDecodeError:
            return {
                "error_rate_threshold": 0.05,
                "p99_latency_threshold_ms": 500,
                "safety_event_threshold": 3,
            }

    def get_compatibility(self) -> dict:
        """Parse compatibility requirements."""
        try:
            return json.loads(self.compatibility_json)
        except json.JSONDecodeError:
            return {}


# =============================================================================
# DeploymentAssignment - Device Assignment to Deployment
# =============================================================================

class DeploymentAssignment(SQLModel, table=True):
    """
    Tracks which devices are assigned to which deployment variant.

    Assignments are deterministic based on device_id hash.
    """
    __tablename__ = "deployment_assignment"
    __table_args__ = (
        UniqueConstraint('deployment_id', 'device_id', name='uq_deployment_device'),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    tenant_id: str = Field(
        foreign_key="tenant.id",
        index=True
    )
    fleet_id: str = Field(
        foreign_key="fleet.id",
        index=True
    )
    device_id: str = Field(
        index=True,
        description="Device identifier (external)"
    )
    deployment_id: str = Field(
        foreign_key="deployment_plan.id",
        index=True
    )

    # Assignment details
    assigned_variant: Optional[str] = Field(
        default=None,
        description="A/B variant (A or B)"
    )
    assigned_adapter_id: Optional[str] = Field(
        default=None,
        description="Assigned adapter ID for this device"
    )
    is_shadow: bool = Field(
        default=False,
        description="Whether running in shadow mode"
    )

    # Bucket assignment (0-9999 for deterministic cohort)
    bucket: int = Field(
        default=0,
        description="Deterministic bucket (0-9999)"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow
    )


# =============================================================================
# RollbackEvent - Rollback Trigger Records
# =============================================================================

class RollbackEvent(SQLModel, table=True):
    """
    Records rollback events for audit and analysis.
    """
    __tablename__ = "rollback_event"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    tenant_id: str = Field(
        foreign_key="tenant.id",
        index=True
    )
    fleet_id: str = Field(
        foreign_key="fleet.id",
        index=True
    )
    deployment_id: str = Field(
        foreign_key="deployment_plan.id",
        index=True
    )

    # Trigger information
    trigger_type: str = Field(
        description="Type of trigger (telemetry_threshold/manual/compat_violation)"
    )
    trigger_details_json: str = Field(
        default="{}",
        description="Details about what triggered the rollback"
    )

    ts: datetime = Field(
        default_factory=datetime.utcnow,
        index=True
    )

    def get_trigger_details(self) -> dict:
        """Parse trigger details."""
        try:
            return json.loads(self.trigger_details_json)
        except json.JSONDecodeError:
            return {}


# =============================================================================
# CompatibilityEvent - Version Compatibility Check Records
# =============================================================================

class CompatibilityEvent(SQLModel, table=True):
    """
    Records compatibility check failures.
    """
    __tablename__ = "compatibility_event"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    tenant_id: str = Field(
        foreign_key="tenant.id",
        index=True
    )
    fleet_id: str = Field(
        foreign_key="fleet.id",
        index=True
    )
    device_id: str = Field(
        index=True
    )
    deployment_id: str = Field(
        foreign_key="deployment_plan.id",
        index=True
    )

    # Failure details
    check_type: str = Field(
        description="Type of check (agent_version/runtime_version/firmware/sensor)"
    )
    required_value: str = Field(
        description="Required version/value"
    )
    actual_value: Optional[str] = Field(
        default=None,
        description="Actual version/value from device"
    )
    passed: bool = Field(
        default=False
    )

    ts: datetime = Field(
        default_factory=datetime.utcnow,
        index=True
    )
