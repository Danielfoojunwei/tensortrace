# TensorGuardFlow Production Hardening Plan

**Version:** 1.0
**Date:** 2026-01-11
**Status:** Implementation Ready
**Audit Scope:** Full System (Frontend + Backend + Database + Crypto)

---

## Executive Summary

This document outlines the comprehensive production hardening plan for TensorGuardFlow v2.3. Based on a complete system audit, we identified **39 issues** across security, reliability, metrics, and VLA/robotics domains. This plan provides a prioritized roadmap to achieve production readiness with empirical, canonical, and unified architecture.

### Key Metrics
| Category | Issues Found | Critical | High | Medium |
|----------|-------------|----------|------|--------|
| Authentication/Authorization | 7 | 2 | 3 | 2 |
| Error Handling | 5 | 1 | 3 | 1 |
| Database Constraints | 11 | 0 | 4 | 7 |
| Input Validation | 4 | 1 | 2 | 1 |
| Metrics Collection | 3 | 0 | 2 | 1 |
| VLA/Robotics Coverage | 5 | 0 | 2 | 3 |
| Security Configuration | 4 | 1 | 2 | 1 |
| **Total** | **39** | **5** | **18** | **16** |

---

## Phase 1: Critical Security Hardening

### 1.1 Authentication Gaps (CRITICAL)

**Problem:** Multiple endpoints lack authentication, exposing sensitive data.

| Endpoint | File:Line | Risk Level |
|----------|-----------|------------|
| `GET /pipeline/config` | `pipeline_config_endpoints.py:23` | HIGH |
| `GET /settings` | `settings_endpoints.py:25` | HIGH |
| `GET /edge/nodes` | `edge_gating_endpoints.py:32` | HIGH |
| `GET /policy/config` | `bayesian_policy_endpoints.py:47` | MEDIUM |
| `GET /lineage/versions` | `lineage_endpoints.py:82` | MEDIUM |
| `GET /integrations/status` | `integrations_endpoints.py:45` | MEDIUM |
| `GET /pipeline/telemetry` | `endpoints.py:209` | MEDIUM |

**Solution:**
```python
# Add to all unprotected endpoints:
from ..auth import get_current_user
from ..models.core import User

@router.get("/endpoint")
async def protected_endpoint(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)  # ADD THIS
):
    # Filter by tenant_id for multi-tenancy
    .where(Model.tenant_id == current_user.tenant_id)
```

**Acceptance Criteria:**
- [ ] All 16 API endpoint files have authentication on sensitive routes
- [ ] Tenant isolation verified on all data-returning endpoints
- [ ] Audit log entries created for sensitive operations

---

### 1.2 DEMO Mode Security Gate (CRITICAL)

**Problem:** `TG_DEMO_MODE=true` bypasses all authentication (auth.py:206-218).

**Current Behavior:**
```python
DEMO_MODE = os.getenv("TG_DEMO_MODE", "true").lower() == "true"
if DEMO_MODE and not token:
    return demo_user  # BYPASSES ALL AUTH
```

**Solution:**
1. Change default to `false` in production builds
2. Add startup warning if DEMO_MODE enabled
3. Add environment-based lockout

```python
# auth.py modification
DEMO_MODE = os.getenv("TG_DEMO_MODE", "false").lower() == "true"
ENVIRONMENT = os.getenv("TG_ENVIRONMENT", "development")

if DEMO_MODE and ENVIRONMENT == "production":
    raise RuntimeError("FATAL: DEMO_MODE cannot be enabled in production")
```

**Acceptance Criteria:**
- [ ] Default `TG_DEMO_MODE=false`
- [ ] Production environment blocks demo mode
- [ ] Startup health check validates auth configuration

---

### 1.3 Rate Limiting Implementation (HIGH)

**Problem:** Rate limiting configured but never implemented (auth.py:60-62).

**Solution:** Implement Redis-backed rate limiting:

```python
# New file: src/tensorguard/platform/middleware/rate_limit.py
from fastapi import Request, HTTPException
from redis import Redis
import os

RATE_LIMIT_ENABLED = os.getenv("TG_ENABLE_RATE_LIMITING", "false").lower() == "true"
REDIS_URL = os.getenv("TG_REDIS_URL", "redis://localhost:6379")

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.redis = Redis.from_url(REDIS_URL) if RATE_LIMIT_ENABLED else None

    async def check(self, request: Request, identifier: str):
        if not self.redis:
            return True
        key = f"rate:{identifier}:{request.url.path}"
        current = self.redis.incr(key)
        if current == 1:
            self.redis.expire(key, 60)
        if current > self.rpm:
            raise HTTPException(429, "Rate limit exceeded")
        return True
```

**Acceptance Criteria:**
- [ ] Rate limiting middleware implemented
- [ ] Login endpoint limited to 5 attempts/15 min
- [ ] API endpoints limited to 100 req/min per user
- [ ] Redis connection pool configured

---

## Phase 2: Reliability Hardening

### 2.1 Database Connection Pooling (HIGH)

**Problem:** No connection pooling configured (database.py:27).

**Current:**
```python
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
```

**Solution:**
```python
# database.py - Production configuration
from sqlalchemy.pool import QueuePool

def create_production_engine(url: str):
    if url.startswith("sqlite"):
        # SQLite: single-threaded
        return create_engine(url, connect_args={"check_same_thread": False})

    # PostgreSQL/MySQL: connection pooling
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_pre_ping=True,  # Validate connections before use
        pool_recycle=3600,   # Recycle connections after 1 hour
        echo_pool="debug" if os.getenv("TG_DEBUG") else False
    )

engine = create_production_engine(DATABASE_URL)

# Session factory for background tasks
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
```

**Acceptance Criteria:**
- [ ] Connection pooling enabled for PostgreSQL
- [ ] Pool pre-ping validates stale connections
- [ ] Pool metrics exposed to monitoring
- [ ] Graceful degradation if pool exhausted

---

### 2.2 Missing Database Indexes (HIGH)

**Problem:** Query performance will degrade at scale without proper indexes.

**Required Indexes:**

```python
# models/core.py
class Fleet(SQLModel, table=True):
    tenant_id: str = Field(foreign_key="tenant.id", index=True)  # ADD index=True
    name: str = Field(index=True)  # ADD index=True
    # Composite index via __table_args__
    __table_args__ = (
        Index('ix_fleet_tenant_active', 'tenant_id', 'is_active'),
    )

class Job(SQLModel, table=True):
    status: str = Field(default="pending", index=True)  # ADD index=True
    __table_args__ = (
        Index('ix_job_fleet_status', 'fleet_id', 'status'),
        Index('ix_job_tenant_created', 'tenant_id', 'created_at'),
    )

# models/fedmoe_models.py
class FedMoEExpert(SQLModel, table=True):
    __table_args__ = (
        Index('ix_expert_tenant_status', 'tenant_id', 'status'),
        UniqueConstraint('tenant_id', 'name', name='uq_expert_tenant_name'),
    )

# models/settings_models.py
class KMSKey(SQLModel, table=True):
    status: str = Field(default="active", index=True)
    tenant_id: Optional[str] = Field(foreign_key="tenant.id", index=True)
```

**Acceptance Criteria:**
- [ ] All foreign keys have indexes
- [ ] Composite indexes for common query patterns
- [ ] Unique constraints prevent duplicate names per tenant
- [ ] Migration script generated for existing data

---

### 2.3 Error Handling Standardization (HIGH)

**Problem:** Inconsistent error handling exposes internal details.

**Bad Patterns Found:**
```python
# endpoints.py:70-74 - Exposes stack trace
except Exception as e:
    traceback.print_exc()  # Leaks to logs
    raise HTTPException(status_code=500, detail=str(e))  # Leaks to client

# lineage_endpoints.py:202 - Silent failure
except:  # Bare except
    return "unknown"
```

**Solution:** Centralized error handling:

```python
# New file: src/tensorguard/platform/middleware/error_handler.py
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import traceback
import uuid

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException:
            raise  # Let FastAPI handle HTTP exceptions
        except Exception as e:
            # Generate error ID for correlation
            error_id = str(uuid.uuid4())[:8]

            # Log full details internally
            logger.error(
                f"Unhandled error {error_id}: {type(e).__name__}: {e}",
                extra={
                    "error_id": error_id,
                    "path": request.url.path,
                    "method": request.method,
                    "traceback": traceback.format_exc()
                }
            )

            # Return safe message to client
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "error_id": error_id,
                    "message": "An unexpected error occurred. Contact support with error_id."
                }
            )
```

**Acceptance Criteria:**
- [ ] No stack traces in API responses
- [ ] All errors have correlation IDs
- [ ] Structured logging with context
- [ ] Sentry/error tracking integration

---

### 2.4 Input Validation Hardening (HIGH)

**Problem:** Query parameters and request bodies lack validation.

**Issues Found:**
- `identity_endpoints.py:420` - Unvalidated enum conversion
- `config_endpoints.py:66` - Unvalidated Dict[str, Any]
- `runs_endpoints.py:56` - Unsanitized file upload

**Solution:** Strict Pydantic models:

```python
# Strict enum validation
from pydantic import validator
from fastapi import Query

class StatusFilter(BaseModel):
    status: Optional[str] = None

    @validator('status')
    def validate_status(cls, v):
        if v and v not in ['pending', 'running', 'completed', 'failed']:
            raise ValueError(f'Invalid status: {v}')
        return v

# File upload validation
from fastapi import UploadFile, File
import re

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.pt', '.pth', '.safetensors', '.onnx'}

async def validate_upload(file: UploadFile):
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")
    await file.seek(0)

    # Sanitize filename
    safe_name = re.sub(r'[^\w\-\.]', '_', file.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Invalid file type: {ext}")

    return safe_name
```

**Acceptance Criteria:**
- [ ] All query parameters have type validation
- [ ] File uploads size-limited and type-checked
- [ ] Filename sanitization prevents path traversal
- [ ] JSON schema validation on complex inputs

---

## Phase 3: Unified API Patterns

### 3.1 Response Format Standardization

**Problem:** Inconsistent response structures across endpoints.

**Current Variations:**
```python
# Pattern A: Wrapped
{"config": {...}}

# Pattern B: Raw list
[{...}, {...}]

# Pattern C: Status object
{"ok": True, "data": [...]}

# Pattern D: Mixed
{"keys": [...], "total": 10}
```

**Canonical Response Format:**

```python
# New file: src/tensorguard/platform/schemas/responses.py
from pydantic import BaseModel
from typing import Generic, TypeVar, Optional, List
from datetime import datetime

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    """Canonical API response wrapper."""
    success: bool = True
    data: T
    meta: Optional[dict] = None
    timestamp: datetime = datetime.utcnow()

class PaginatedResponse(APIResponse[List[T]], Generic[T]):
    """Paginated list response."""
    meta: dict = {
        "page": 1,
        "limit": 100,
        "total": 0,
        "has_more": False
    }

# Usage example:
@router.get("/fleets", response_model=PaginatedResponse[FleetSchema])
async def list_fleets(...):
    return PaginatedResponse(
        data=fleets,
        meta={"page": page, "limit": limit, "total": total, "has_more": has_more}
    )
```

**Acceptance Criteria:**
- [ ] All endpoints use `APIResponse` wrapper
- [ ] Paginated endpoints use `PaginatedResponse`
- [ ] OpenAPI schema reflects canonical format
- [ ] Frontend updated to parse canonical format

---

### 3.2 Canonical Metrics Collection

**Problem:** Metrics are mock/random data, not empirical measurements.

**Mock Data Found:**
- `endpoints.py:82-116` - `random.randint()` for device counts
- `forensics_endpoints.py:132-192` - Hardcoded latency values
- `endpoints.py:208-256` - Simulated telemetry

**Solution:** Real metrics infrastructure:

```python
# New file: src/tensorguard/platform/metrics/collector.py
from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time

# Standard metrics
REQUEST_COUNT = Counter(
    'tg_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'tg_request_latency_seconds',
    'Request latency',
    ['method', 'endpoint'],
    buckets=[.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10]
)

ACTIVE_CONNECTIONS = Gauge(
    'tg_active_connections',
    'Active database connections'
)

# VLA-specific metrics
VLA_INFERENCE_LATENCY = Histogram(
    'tg_vla_inference_seconds',
    'VLA model inference latency',
    ['model', 'task_type'],
    buckets=[.01, .025, .05, .1, .25, .5, 1, 2]
)

EXPERT_THROUGHPUT = Counter(
    'tg_expert_throughput_total',
    'Expert inference throughput',
    ['expert_name', 'fleet_id']
)

# Middleware for automatic collection
def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(latency)

    return response
```

**New `/metrics` Endpoint:**
```python
# main.py addition
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

@app.get("/metrics")
async def prometheus_metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

**Acceptance Criteria:**
- [ ] Prometheus metrics endpoint at `/metrics`
- [ ] Request latency histograms for all endpoints
- [ ] Database connection pool metrics
- [ ] VLA-specific inference metrics
- [ ] No mock/random data in production metrics

---

### 3.3 Replace Mock Data with Real Queries

**Files Requiring Updates:**

| File | Mock Pattern | Real Implementation |
|------|-------------|---------------------|
| `endpoints.py:82-116` | `random.randint()` | Count from Device table |
| `endpoints.py:208-256` | Hardcoded telemetry | Query from metrics store |
| `forensics_endpoints.py:54-75` | Mock incidents | Query from Incident table |
| `forensics_endpoints.py:132-192` | Random latency | Aggregate from metrics |
| `edge_gating_endpoints.py:25-30` | In-memory nodes | Query from EdgeNode table |

**Example Migration:**

```python
# BEFORE (endpoints.py:82-116)
devices_total = random.randint(50, 500)
trust_score = random.uniform(85.0, 99.9)

# AFTER
from ..models.core import Device
from sqlmodel import func

@router.get("/fleets/extended")
async def get_fleets_extended(session: Session, current_user: User):
    fleets = session.exec(
        select(Fleet).where(Fleet.tenant_id == current_user.tenant_id)
    ).all()

    result = []
    for fleet in fleets:
        # Real device count
        device_count = session.exec(
            select(func.count(Device.id)).where(Device.fleet_id == fleet.id)
        ).one()

        # Real trust score from attestation records
        trust_score = session.exec(
            select(func.avg(Attestation.score))
            .where(Attestation.fleet_id == fleet.id)
            .where(Attestation.timestamp > datetime.utcnow() - timedelta(hours=24))
        ).one() or 0.0

        result.append({
            "fleet": fleet,
            "devices_total": device_count,
            "trust_score": trust_score
        })

    return result
```

**Acceptance Criteria:**
- [ ] Zero `random.` calls in production API code
- [ ] All metrics derived from database or metrics store
- [ ] Fallback to "No data" rather than fake data
- [ ] Metrics timestamps reflect actual collection time

---

## Phase 4: VLA/Robotics Optimization

### 4.1 VLA Model Registry

**Problem:** No dedicated storage for Vision-Language-Action models.

**Solution:** VLA-specific model registry:

```python
# New file: src/tensorguard/platform/models/vla_models.py
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum
import uuid

class VLATaskType(str, Enum):
    PICK_AND_PLACE = "pick_and_place"
    WELDING = "welding"
    INSPECTION = "inspection"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    ASSEMBLY = "assembly"

class VLAModel(SQLModel, table=True):
    """Vision-Language-Action model registry entry."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str = Field(index=True)
    version: str
    task_types: str  # JSON array of VLATaskType values

    # Model architecture
    vision_encoder: str = Field(default="ViT-L/14")
    language_model: str = Field(default="Llama-3-8B")
    action_head: str = Field(default="Diffusion-Policy")

    # Performance metrics
    success_rate: float = Field(default=0.0)
    avg_latency_ms: float = Field(default=0.0)
    safety_score: float = Field(default=0.0)

    # Deployment
    tenant_id: str = Field(foreign_key="tenant.id", index=True)
    fleet_ids: Optional[str] = None  # JSON array of fleet IDs
    status: str = Field(default="staged")  # staged, deployed, deprecated

    created_at: datetime = Field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = None

    # PQC signature for model integrity
    model_hash: str  # SHA-256 of model weights
    pqc_signature: Optional[str] = None


class VLASafetyCheck(SQLModel, table=True):
    """Safety validation records for VLA deployments."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    model_id: str = Field(foreign_key="vlamodel.id", index=True)

    # Safety dimensions
    collision_free_rate: float
    force_limit_compliance: float
    emergency_stop_latency_ms: float
    workspace_boundary_adherence: float

    # Test conditions
    test_environment: str  # simulation, staging, production
    test_scenarios: int
    passed_scenarios: int

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    signed_by: str
    pqc_signature: Optional[str] = None
```

**Acceptance Criteria:**
- [ ] VLA model registry with versioning
- [ ] Task type classification for robotics
- [ ] Safety score tracking per model
- [ ] PQC-signed model integrity verification

---

### 4.2 VLA Safety Framework

**Problem:** No safety validation for robotic commands.

**Solution:** Multi-layer safety checks:

```python
# New file: src/tensorguard/platform/api/vla_safety_endpoints.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

class RobotCommand(BaseModel):
    fleet_id: str
    robot_id: str
    action: str
    parameters: dict

class SafetyValidationResult(BaseModel):
    approved: bool
    checks: List[dict]
    risk_score: float
    recommendations: List[str]

@router.post("/vla/safety/validate")
async def validate_robot_command(
    command: RobotCommand,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
) -> SafetyValidationResult:
    """
    Validate a robot command against safety policies.

    Safety Checks:
    1. Workspace boundary validation
    2. Force/torque limits
    3. Collision prediction
    4. Human proximity detection
    5. Emergency stop availability
    """
    checks = []

    # 1. Workspace boundary check
    workspace_check = await validate_workspace_bounds(command)
    checks.append({
        "name": "Workspace Boundary",
        "status": "PASS" if workspace_check else "FAIL",
        "details": "Action within defined workspace"
    })

    # 2. Force limit check
    force_check = await validate_force_limits(command)
    checks.append({
        "name": "Force Limits",
        "status": "PASS" if force_check else "FAIL",
        "details": f"Max force within {MAX_FORCE_N}N limit"
    })

    # 3. Collision prediction
    collision_risk = await predict_collision(command)
    checks.append({
        "name": "Collision Prediction",
        "status": "PASS" if collision_risk < 0.1 else "WARN" if collision_risk < 0.5 else "FAIL",
        "details": f"Collision probability: {collision_risk:.2%}"
    })

    # Calculate overall approval
    failed_checks = [c for c in checks if c["status"] == "FAIL"]
    risk_score = len(failed_checks) / len(checks)

    return SafetyValidationResult(
        approved=len(failed_checks) == 0,
        checks=checks,
        risk_score=risk_score,
        recommendations=generate_safety_recommendations(checks)
    )

@router.get("/vla/safety/metrics")
async def get_safety_metrics(
    fleet_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Get aggregated safety metrics for a fleet."""
    # Query VLASafetyCheck records
    safety_records = session.exec(
        select(VLASafetyCheck)
        .join(VLAModel)
        .where(VLAModel.fleet_ids.contains(fleet_id))
        .order_by(VLASafetyCheck.timestamp.desc())
        .limit(100)
    ).all()

    return {
        "fleet_id": fleet_id,
        "total_validations": len(safety_records),
        "avg_collision_free_rate": avg([r.collision_free_rate for r in safety_records]),
        "avg_force_compliance": avg([r.force_limit_compliance for r in safety_records]),
        "avg_estop_latency_ms": avg([r.emergency_stop_latency_ms for r in safety_records]),
        "trend": calculate_safety_trend(safety_records)
    }
```

**Acceptance Criteria:**
- [ ] Pre-execution safety validation
- [ ] Real-time collision prediction
- [ ] Force/torque limit enforcement
- [ ] Safety audit trail with PQC signatures
- [ ] Fleet-wide safety dashboards

---

### 4.3 VLA PEFT Optimization

**Problem:** PEFT training not optimized for VLA models.

**Solution:** VLA-specific training configuration:

```python
# Enhanced PeftWizardState for VLA
class VLAPeftConfig(BaseModel):
    """VLA-specific PEFT configuration."""

    # Base model selection
    vision_encoder: str = "ViT-L/14"
    language_model: str = "Llama-3-8B"
    action_head: str = "Diffusion-Policy"

    # LoRA configuration per component
    lora_config: dict = {
        "vision": {"r": 16, "alpha": 32, "dropout": 0.05},
        "language": {"r": 64, "alpha": 128, "dropout": 0.1},
        "action": {"r": 32, "alpha": 64, "dropout": 0.05}
    }

    # VLA-specific training
    action_prediction_horizon: int = 16
    proprioception_dim: int = 14  # Joint angles + gripper
    action_dim: int = 7  # 6-DOF + gripper

    # Safety constraints during training
    safety_margin: float = 0.1
    max_action_magnitude: float = 0.1  # Normalized

    # Evaluation metrics
    eval_tasks: List[str] = ["pick_and_place", "navigation", "manipulation"]
    eval_episodes: int = 100
    success_threshold: float = 0.9

# Training workflow enhancement
@router.post("/peft/vla/runs")
async def start_vla_training(
    config: VLAPeftConfig,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Start VLA-optimized PEFT training run."""

    # Validate safety constraints
    if config.max_action_magnitude > 0.2:
        raise HTTPException(400, "Action magnitude exceeds safety limit")

    # Create training run with VLA-specific stages
    run = PeftRun(
        tenant_id=current_user.tenant_id,
        created_by_user_id=current_user.id,
        config_json=config.dict(),
        status=PeftRunStatus.PENDING,
        stage="VLA_INIT",
        vla_mode=True
    )
    session.add(run)
    session.commit()

    # Start VLA workflow
    background_tasks.add_task(run_vla_training, run.id)

    return {"run_id": run.id, "status": "pending", "mode": "vla"}
```

**Acceptance Criteria:**
- [ ] VLA-specific model configuration
- [ ] Per-component LoRA configuration (vision/language/action)
- [ ] Safety constraints enforced during training
- [ ] VLA benchmark evaluation (1000 cycles, 5 tasks)
- [ ] Success rate tracking with canonical metrics

---

## Phase 5: Production Readiness Checklist

### 5.1 Environment Configuration

```bash
# Required environment variables for production
TG_SECRET_KEY=<generated-256-bit-key>
TG_DEMO_MODE=false
TG_ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@host:5432/tensorguard
TG_REDIS_URL=redis://host:6379/0
TG_ENABLE_RATE_LIMITING=true
TG_TOKEN_EXPIRE_MINUTES=30
TG_MIN_PASSWORD_LENGTH=12
TG_REQUIRE_PASSWORD_COMPLEXITY=true
```

### 5.2 Security Headers

```python
# main.py - Add security headers middleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*.tensorguard.io", "localhost"]
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response
```

### 5.3 Health Check Endpoints

```python
# New endpoint: /health
@app.get("/health")
async def health_check(session: Session = Depends(get_session)):
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }

    # Database connectivity
    try:
        session.exec(select(1)).first()
        checks["checks"]["database"] = "ok"
    except Exception:
        checks["checks"]["database"] = "error"
        checks["status"] = "unhealthy"

    # Redis connectivity
    if RATE_LIMIT_ENABLED:
        try:
            redis.ping()
            checks["checks"]["redis"] = "ok"
        except Exception:
            checks["checks"]["redis"] = "error"
            checks["status"] = "degraded"

    return checks

@app.get("/ready")
async def readiness_check(session: Session = Depends(get_session)):
    """Kubernetes readiness probe."""
    try:
        session.exec(select(1)).first()
        return {"ready": True}
    except Exception:
        raise HTTPException(503, "Not ready")
```

### 5.4 Deployment Verification

| Check | Command | Expected |
|-------|---------|----------|
| Auth disabled | `TG_DEMO_MODE` | `false` |
| Rate limiting | `TG_ENABLE_RATE_LIMITING` | `true` |
| HTTPS redirect | Request HTTP | 301 to HTTPS |
| Security headers | Check response | All present |
| Database pool | Monitor connections | Pool metrics exposed |
| Metrics endpoint | `GET /metrics` | Prometheus format |
| Health check | `GET /health` | `{"status": "healthy"}` |

---

## Implementation Schedule

### Iteration 1: Critical Security (Immediate)
- [ ] Add authentication to all endpoints
- [ ] Disable DEMO_MODE by default
- [ ] Implement rate limiting
- [ ] Add security headers

### Iteration 2: Reliability (Week 1-2)
- [ ] Database connection pooling
- [ ] Add missing indexes
- [ ] Standardize error handling
- [ ] Input validation hardening

### Iteration 3: Canonical Metrics (Week 2-3)
- [ ] Prometheus metrics integration
- [ ] Replace mock data with real queries
- [ ] Standardize response formats
- [ ] Add health check endpoints

### Iteration 4: VLA Optimization (Week 3-4)
- [ ] VLA model registry
- [ ] Safety validation framework
- [ ] VLA PEFT configuration
- [ ] Robotics dashboard

### Iteration 5: Production Deployment (Week 4+)
- [ ] Load testing (1000 req/s target)
- [ ] Security penetration testing
- [ ] Disaster recovery testing
- [ ] Documentation completion

---

## Appendix: Files Modified

### Backend Changes
| File | Changes |
|------|---------|
| `auth.py` | DEMO_MODE default, rate limiting |
| `database.py` | Connection pooling |
| `main.py` | Security middleware, health endpoints |
| `models/core.py` | Indexes, constraints |
| `models/vla_models.py` | NEW - VLA registry |
| All `*_endpoints.py` | Auth, error handling, validation |

### Frontend Changes
| File | Changes |
|------|---------|
| `stores/*.js` | Canonical response parsing |
| `components/*.vue` | Error handling, loading states |

### New Files
| File | Purpose |
|------|---------|
| `middleware/rate_limit.py` | Rate limiting |
| `middleware/error_handler.py` | Centralized errors |
| `metrics/collector.py` | Prometheus metrics |
| `models/vla_models.py` | VLA model registry |
| `api/vla_safety_endpoints.py` | Safety framework |
| `schemas/responses.py` | Canonical responses |

---

**Document Status:** Ready for Implementation
**Next Action:** Phase 1 - Critical Security Hardening
