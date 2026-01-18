"""
System Settings Model for TensorGuard Platform.
Provides persistent storage for global platform configuration.
"""

from sqlmodel import SQLModel, Field
from sqlalchemy import Index
from typing import Optional
from datetime import datetime
from enum import Enum
import uuid


class KeyStatus(str, Enum):
    """Canonical KMS key lifecycle states."""
    ACTIVE = "active"
    ROTATING = "rotating"
    REVOKED = "revoked"
    EXPIRED = "expired"


class RotationAction(str, Enum):
    """Canonical key rotation actions."""
    CREATED = "created"
    ROTATED = "rotated"
    REVOKED = "revoked"
    EXPIRED = "expired"


class SystemSettingBase(SQLModel):
    key: str = Field(index=True, unique=True)
    value: str
    description: Optional[str] = None


class SystemSetting(SystemSettingBase, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None
    tenant_id: Optional[str] = Field(default=None, foreign_key="tenant.id", index=True)


class FleetPolicyRecord(SQLModel, table=True):
    """
    Persisted fleet policy configuration.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id", index=True)
    fleet_id: str = Field(foreign_key="fleet.id", index=True, unique=True)
    policy_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = Field(default=None, foreign_key="user.id")


class KMSKey(SQLModel, table=True):
    """
    Managed cryptographic key for the TensorGuard KMS.
    Persists key metadata (not the actual key material for security).
    """
    __table_args__ = (
        Index('ix_kms_tenant_status', 'tenant_id', 'status'),
        Index('ix_kms_rotation_due', 'status', 'last_rotated_at'),
    )

    kid: str = Field(primary_key=True)  # Key ID
    region: str = Field(default="global", index=True)
    algorithm: str = Field(default="Kyber-768 + Ed25519")
    status: str = Field(default=KeyStatus.ACTIVE.value, index=True)
    rotation_ttl_days: int = Field(default=30)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    last_rotated_at: Optional[datetime] = None
    next_rotation_at: Optional[datetime] = None  # Pre-computed rotation deadline
    tenant_id: Optional[str] = Field(default=None, foreign_key="tenant.id", index=True)

    # Key usage tracking
    usage_count: int = Field(default=0)
    last_used_at: Optional[datetime] = None


class KMSRotationLog(SQLModel, table=True):
    """Immutable log of key rotation events."""
    __table_args__ = (
        Index('ix_rotation_kid_timestamp', 'kid', 'timestamp'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    kid: str = Field(index=True, foreign_key="kmskey.kid")  # Foreign key to KMSKey
    action: str = Field(index=True)  # Uses RotationAction values
    reason: Optional[str] = None
    performed_by: Optional[str] = Field(default=None, foreign_key="user.id")
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    pqc_signature: Optional[str] = None  # Dilithium-3 signature for tamper-evidence


# ============================================================================
# Edge Gating Models (Production Hardened)
# ============================================================================

class EdgeNodeStatus(str, Enum):
    """Edge node connection status."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"


class EdgeNode(SQLModel, table=True):
    """
    Registered edge node for gating control.
    Persists node configuration and connection state.
    """
    __table_args__ = (
        Index('ix_edge_node_tenant_status', 'tenant_id', 'status'),
        Index('ix_edge_node_fleet', 'fleet_id'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    node_id: str = Field(unique=True, index=True)  # External node identifier
    tenant_id: str = Field(foreign_key="tenant.id", index=True)
    fleet_id: Optional[str] = Field(default=None, foreign_key="fleet.id", index=True)

    # Gating configuration
    gating_enabled: bool = Field(default=True)
    local_threshold: float = Field(default=0.15)
    task_whitelist: str = Field(default="[]")  # JSON array of allowed tasks

    # Connection state
    status: str = Field(default=EdgeNodeStatus.OFFLINE.value, index=True)
    last_heartbeat: Optional[datetime] = None
    last_ip_address: Optional[str] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TelemetrySample(SQLModel, table=True):
    """
    Telemetry sample from edge node gating decisions.
    Real telemetry data POSTed by edge agents.
    """
    __table_args__ = (
        Index('ix_telemetry_node_time', 'node_id', 'timestamp'),
        Index('ix_telemetry_tenant_time', 'tenant_id', 'timestamp'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    node_id: str = Field(index=True, foreign_key="edgenode.id")
    tenant_id: str = Field(foreign_key="tenant.id", index=True)

    # Gating decision data
    task: str = Field(index=True)
    relevance_score: float
    threshold: float
    decision: str  # PASS, BLOCK
    latency_ms: Optional[float] = None

    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)


class GatingDecisionLog(SQLModel, table=True):
    """
    Aggregate gating decision log for analytics.
    Summarizes decisions over time windows.
    """
    __table_args__ = (
        Index('ix_gating_log_node_window', 'node_id', 'window_start'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    node_id: str = Field(index=True, foreign_key="edgenode.id")
    tenant_id: str = Field(foreign_key="tenant.id", index=True)

    window_start: datetime = Field(index=True)
    window_end: datetime
    total_decisions: int = Field(default=0)
    pass_count: int = Field(default=0)
    block_count: int = Field(default=0)
    avg_relevance_score: float = Field(default=0.0)
    avg_latency_ms: float = Field(default=0.0)


# ============================================================================
# Integration Connection Models (Production Hardened)
# ============================================================================

class IntegrationStatus(str, Enum):
    """Integration connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class IntegrationConnection(SQLModel, table=True):
    """
    Persisted integration connection state.
    Tracks real connection status and health.
    """
    __table_args__ = (
        Index('ix_integration_tenant_service', 'tenant_id', 'service'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id", index=True)
    service: str = Field(index=True)  # isaac_lab, ros2_bridge, formant, huggingface

    # Connection state
    status: str = Field(default=IntegrationStatus.DISCONNECTED.value, index=True)
    config_json: str = Field(default="{}")  # Encrypted config
    last_seen: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_check_latency_ms: Optional[float] = None
    error_message: Optional[str] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
