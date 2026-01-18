from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Index, UniqueConstraint
from datetime import datetime
from enum import Enum
import uuid


class UserRole(str, Enum):
    ORG_ADMIN = "org_admin"
    SITE_ADMIN = "site_admin"
    OPERATOR = "operator"
    AUDITOR = "auditor"
    SERVICE_ACCOUNT = "service_account"


class JobType(str, Enum):
    """Canonical job types for type safety."""
    TRAIN = "TRAIN"
    EVAL = "EVAL"
    DEPLOY = "DEPLOY"
    VLA_TRAIN = "VLA_TRAIN"
    VLA_EVAL = "VLA_EVAL"


class JobStatus(str, Enum):
    """Canonical job statuses for type safety."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Tenant(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str = Field(index=True, unique=True)  # Tenant names must be unique
    plan: str = Field(default="starter")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    users: List["User"] = Relationship(back_populates="tenant")
    fleets: List["Fleet"] = Relationship(back_populates="tenant")


class User(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    email: str = Field(unique=True, index=True)
    name: Optional[str] = None  # Display name
    hashed_password: str
    role: UserRole = Field(default=UserRole.OPERATOR)
    tenant_id: str = Field(foreign_key="tenant.id", index=True)
    is_active: bool = Field(default=True)  # For account deactivation

    tenant: Tenant = Relationship(back_populates="users")


class Fleet(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint('tenant_id', 'name', name='uq_fleet_tenant_name'),
        Index('ix_fleet_tenant_active', 'tenant_id', 'is_active'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str = Field(index=True)
    tenant_id: str = Field(foreign_key="tenant.id", index=True)
    api_key_hash: str
    is_active: bool = Field(default=True)
    region: Optional[str] = Field(default=None, index=True)  # For regional queries

    tenant: Tenant = Relationship(back_populates="fleets")
    jobs: List["Job"] = Relationship(back_populates="fleet")


class Job(SQLModel, table=True):
    __table_args__ = (
        Index('ix_job_fleet_status', 'fleet_id', 'status'),
        Index('ix_job_status_created', 'status', 'created_at'),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    fleet_id: str = Field(foreign_key="fleet.id", index=True)
    type: str = Field(index=True)  # Uses JobType values
    status: str = Field(default=JobStatus.PENDING.value, index=True)
    config_json: str = Field(default="{}")
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    completed_at: Optional[datetime] = None

    fleet: Fleet = Relationship(back_populates="jobs")

class AuditLog(SQLModel, table=True):
    """Traceability ledger for SOC 2 and ISO 9001 compliance."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id", index=True)
    user_id: Optional[str] = Field(foreign_key="user.id", nullable=True)
    action: str  # e.g., "KEY_SIGN", "PACKAGE_UPLOAD", "MODEL_DEPLOY"
    resource_id: str
    resource_type: str
    details: str = Field(default="{}") # JSON blob
    pqc_signature: Optional[str] = None # Dilithium-3 hex signature
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    success: bool = True

class ReplayNonce(SQLModel, table=True):
    """Store nonces to prevent HMAC replay attacks."""
    nonce: str = Field(primary_key=True)
    fleet_id: str = Field(index=True)
    timestamp: int = Field(index=True)
    expires_at: datetime = Field(index=True)
