"""
Production Telemetry Ingestion & Query API

Replaces all simulated telemetry endpoints with real database-backed
ingestion and aggregation. All data is persisted and queryable.

Key endpoints:
- POST /telemetry/ingest: Batch ingestion from edge agents (HMAC auth)
- GET /telemetry/pipeline: Aggregated pipeline metrics (user auth)
- GET /telemetry/edge: Latest gating stage events per device (user auth)

No random/mock data in production paths.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlmodel import Session, select, func, col

from ..database import get_session
from ..auth import get_current_user
from ..models.core import User, Fleet
from ..models.telemetry_models import (
    FleetDevice,
    TelemetryStageEvent,
    TelemetrySystemEvent,
    TelemetryModelBehaviorEvent,
    ForensicsEvent,
    PipelineStage,
    StageStatus,
    EventSeverity,
)
from .identity_endpoints import verify_fleet_auth

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class TelemetryMessage(BaseModel):
    """Single telemetry message from edge agent."""
    topic: str = Field(..., description="Topic: telemetry.stage|system|model_behavior|forensics")
    timestamp_ns: int = Field(..., description="Event timestamp in nanoseconds")
    payload: Dict[str, Any] = Field(..., description="Event payload")
    priority: int = Field(default=0, description="Message priority (0=normal)")


class DeviceInfo(BaseModel):
    """Device information for upsert."""
    device_id: str
    agent_version: Optional[str] = None
    runtime_version: Optional[str] = None
    ros_distro: Optional[str] = None
    firmware_version: Optional[str] = None
    sensor_manifest_hash: Optional[str] = None


class TelemetryBatch(BaseModel):
    """Batch of telemetry messages from edge agent."""
    batch_id: str = Field(..., description="Unique batch identifier")
    device_info: Optional[DeviceInfo] = Field(None, description="Device information for upsert")
    messages: List[TelemetryMessage] = Field(..., description="Telemetry messages")


class IngestResult(BaseModel):
    """Result of telemetry ingestion."""
    accepted: int
    rejected: int
    rejections: List[Dict[str, Any]] = []


class StageMetrics(BaseModel):
    """Aggregated metrics for a pipeline stage."""
    stage: str
    status: str
    count: int
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    degraded_rate: float


class PipelineMetrics(BaseModel):
    """Complete pipeline telemetry response."""
    fleet_id: str
    timestamp: str
    safe_mode: bool
    workflow: List[Dict[str, Any]]
    summary: Dict[str, Any]


# =============================================================================
# Telemetry Ingestion (Edge Agent Auth)
# =============================================================================

@router.post("/telemetry/ingest", response_model=IngestResult)
async def ingest_telemetry(
    batch: TelemetryBatch,
    session: Session = Depends(get_session),
    fleet: Fleet = Depends(verify_fleet_auth),
):
    """
    Ingest telemetry batch from edge agents.

    Auth: HMAC signature via verify_fleet_auth (X-TG-Fleet-Id, X-TG-Timestamp,
          X-TG-Nonce, X-TG-Signature headers).

    Accepts batches with topics:
    - telemetry.stage: Pipeline stage events
    - telemetry.system: System resource events
    - telemetry.model_behavior: Model decision events
    - telemetry.forensics: Safety/security events

    Returns accepted/rejected counts with rejection reasons.
    """
    accepted = 0
    rejected = 0
    rejections = []

    tenant_id = fleet.tenant_id
    fleet_id = fleet.id

    # Upsert device info if provided
    device_id = None
    if batch.device_info:
        device_id = batch.device_info.device_id
        device = session.exec(
            select(FleetDevice).where(FleetDevice.device_id == device_id)
        ).first()

        if device:
            # Update existing device
            device.agent_version = batch.device_info.agent_version or device.agent_version
            device.runtime_version = batch.device_info.runtime_version or device.runtime_version
            device.ros_distro = batch.device_info.ros_distro or device.ros_distro
            device.firmware_version = batch.device_info.firmware_version or device.firmware_version
            device.sensor_manifest_hash = batch.device_info.sensor_manifest_hash or device.sensor_manifest_hash
            device.last_seen_at = datetime.utcnow()
            device.updated_at = datetime.utcnow()
        else:
            # Create new device
            device = FleetDevice(
                device_id=device_id,
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                agent_version=batch.device_info.agent_version,
                runtime_version=batch.device_info.runtime_version,
                ros_distro=batch.device_info.ros_distro,
                firmware_version=batch.device_info.firmware_version,
                sensor_manifest_hash=batch.device_info.sensor_manifest_hash,
                last_seen_at=datetime.utcnow(),
            )
            session.add(device)

    # Process messages
    for idx, msg in enumerate(batch.messages):
        try:
            ts = datetime.utcfromtimestamp(msg.timestamp_ns / 1_000_000_000)
            msg_device_id = msg.payload.get("device_id", device_id)

            if not msg_device_id:
                rejections.append({
                    "index": idx,
                    "reason": "missing device_id",
                    "topic": msg.topic
                })
                rejected += 1
                continue

            if msg.topic == "telemetry.stage":
                event = _parse_stage_event(tenant_id, fleet_id, msg_device_id, ts, msg.payload)
                if event:
                    session.add(event)
                    accepted += 1
                else:
                    rejections.append({"index": idx, "reason": "invalid stage payload", "topic": msg.topic})
                    rejected += 1

            elif msg.topic == "telemetry.system":
                event = _parse_system_event(tenant_id, fleet_id, msg_device_id, ts, msg.payload)
                if event:
                    session.add(event)
                    accepted += 1
                else:
                    rejections.append({"index": idx, "reason": "invalid system payload", "topic": msg.topic})
                    rejected += 1

            elif msg.topic == "telemetry.model_behavior":
                event = _parse_model_behavior_event(tenant_id, fleet_id, msg_device_id, ts, msg.payload)
                if event:
                    session.add(event)
                    accepted += 1
                else:
                    rejections.append({"index": idx, "reason": "invalid model_behavior payload", "topic": msg.topic})
                    rejected += 1

            elif msg.topic == "telemetry.forensics":
                event = _parse_forensics_event(tenant_id, fleet_id, msg_device_id, ts, msg.payload)
                if event:
                    session.add(event)
                    accepted += 1
                else:
                    rejections.append({"index": idx, "reason": "invalid forensics payload", "topic": msg.topic})
                    rejected += 1

            else:
                rejections.append({"index": idx, "reason": f"unknown topic: {msg.topic}", "topic": msg.topic})
                rejected += 1

        except Exception as e:
            logger.error(f"Error processing message {idx}: {e}")
            rejections.append({"index": idx, "reason": str(e), "topic": msg.topic})
            rejected += 1

    session.commit()

    logger.info(f"Telemetry batch {batch.batch_id}: accepted={accepted}, rejected={rejected}")

    return IngestResult(accepted=accepted, rejected=rejected, rejections=rejections)


def _parse_stage_event(tenant_id: str, fleet_id: str, device_id: str, ts: datetime, payload: Dict) -> Optional[TelemetryStageEvent]:
    """Parse and validate stage event payload."""
    try:
        stage = payload.get("stage")
        if not stage:
            return None

        return TelemetryStageEvent(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            device_id=device_id,
            run_id=payload.get("run_id"),
            stage=stage,
            status=payload.get("status", StageStatus.OK.value),
            latency_ms=float(payload.get("latency_ms", 0)),
            metadata_json=json.dumps(payload.get("metadata", {})),
            ts=ts,
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid stage payload: {e}")
        return None


def _parse_system_event(tenant_id: str, fleet_id: str, device_id: str, ts: datetime, payload: Dict) -> Optional[TelemetrySystemEvent]:
    """Parse and validate system event payload."""
    try:
        return TelemetrySystemEvent(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            device_id=device_id,
            cpu_pct=float(payload.get("cpu_pct", 0)),
            mem_pct=float(payload.get("mem_pct", 0)),
            gpu_pct=float(payload["gpu_pct"]) if payload.get("gpu_pct") is not None else None,
            temp_c=float(payload["temp_c"]) if payload.get("temp_c") is not None else None,
            bandwidth_up_bps=int(payload["bandwidth_up_bps"]) if payload.get("bandwidth_up_bps") is not None else None,
            bandwidth_down_bps=int(payload["bandwidth_down_bps"]) if payload.get("bandwidth_down_bps") is not None else None,
            dropped_frames=int(payload.get("dropped_frames", 0)),
            queue_latency_ms=float(payload["queue_latency_ms"]) if payload.get("queue_latency_ms") is not None else None,
            ts=ts,
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid system payload: {e}")
        return None


def _parse_model_behavior_event(tenant_id: str, fleet_id: str, device_id: str, ts: datetime, payload: Dict) -> Optional[TelemetryModelBehaviorEvent]:
    """Parse and validate model behavior event payload."""
    try:
        model_version = payload.get("model_version")
        decision_hash = payload.get("decision_hash")
        if not model_version or not decision_hash:
            return None

        return TelemetryModelBehaviorEvent(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            device_id=device_id,
            model_version=model_version,
            adapter_id=payload.get("adapter_id"),
            decision_hash=decision_hash,
            action_distribution_json=json.dumps(payload["action_distribution"]) if payload.get("action_distribution") else None,
            refusal_rate=float(payload["refusal_rate"]) if payload.get("refusal_rate") is not None else None,
            tool_call_failures=int(payload.get("tool_call_failures", 0)),
            policy_constraint_hits=int(payload.get("policy_constraint_hits", 0)),
            is_shadow=bool(payload.get("is_shadow", False)),
            ts=ts,
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid model_behavior payload: {e}")
        return None


def _parse_forensics_event(tenant_id: str, fleet_id: str, device_id: str, ts: datetime, payload: Dict) -> Optional[ForensicsEvent]:
    """Parse and validate forensics event payload."""
    try:
        severity = payload.get("severity")
        event_type = payload.get("event_type")
        if not severity or not event_type:
            return None

        return ForensicsEvent(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            device_id=device_id,
            severity=severity,
            event_type=event_type,
            evidence_ref=payload.get("evidence_ref"),
            details_json=json.dumps(payload.get("details", {})),
            ts=ts,
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid forensics payload: {e}")
        return None


# =============================================================================
# Pipeline Telemetry Query (User Auth)
# =============================================================================

@router.get("/telemetry/pipeline")
async def get_pipeline_telemetry(
    fleet_id: Optional[str] = None,
    time_range: str = Query(default="15m", regex="^(15m|1h|24h)$"),
    stage: Optional[str] = None,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get aggregated pipeline telemetry metrics.

    Auth: Bearer token (get_current_user)

    Computes from real telemetry_stage_event data:
    - p50/p90/p99 latency per stage
    - degraded/error rates per stage
    - safe_mode derived from error rates crossing thresholds

    Replaces the simulated endpoint in endpoints.py.
    """
    # Parse time range
    minutes = {"15m": 15, "1h": 60, "24h": 1440}[time_range]
    since = datetime.utcnow() - timedelta(minutes=minutes)

    # Build query
    query = select(TelemetryStageEvent).where(
        TelemetryStageEvent.tenant_id == current_user.tenant_id,
        TelemetryStageEvent.ts >= since,
    )

    if fleet_id:
        # Verify fleet belongs to tenant
        fleet = session.get(Fleet, fleet_id)
        if not fleet or fleet.tenant_id != current_user.tenant_id:
            raise HTTPException(status_code=404, detail="Fleet not found")
        query = query.where(TelemetryStageEvent.fleet_id == fleet_id)
    else:
        # Get first fleet for tenant if not specified
        fleets = session.exec(
            select(Fleet).where(Fleet.tenant_id == current_user.tenant_id).limit(1)
        ).all()
        if fleets:
            fleet_id = fleets[0].id
            query = query.where(TelemetryStageEvent.fleet_id == fleet_id)
        else:
            fleet_id = "no_fleet"

    if stage:
        query = query.where(TelemetryStageEvent.stage == stage)

    events = session.exec(query.order_by(TelemetryStageEvent.ts.desc()).limit(10000)).all()

    # Aggregate by stage
    stages = ["capture", "embed", "gate", "peft", "shield", "sync", "pull"]
    workflow = []
    total_error_rate = 0.0
    stage_count = 0

    for s in stages:
        stage_events = [e for e in events if e.stage == s]

        if not stage_events:
            workflow.append({
                "stage": s,
                "status": "unknown",
                "latency_ms": 0,
                "metadata": {},
                "metrics": {
                    "count": 0,
                    "p50_latency_ms": 0,
                    "p90_latency_ms": 0,
                    "p99_latency_ms": 0,
                    "error_rate": 0,
                    "degraded_rate": 0,
                }
            })
            continue

        latencies = sorted([e.latency_ms for e in stage_events])
        count = len(latencies)

        p50 = latencies[int(count * 0.50)] if count > 0 else 0
        p90 = latencies[int(count * 0.90)] if count > 0 else 0
        p99 = latencies[int(count * 0.99)] if count > 0 else 0

        error_count = sum(1 for e in stage_events if e.status == StageStatus.ERROR.value)
        degraded_count = sum(1 for e in stage_events if e.status == StageStatus.DEGRADED.value)

        error_rate = error_count / count if count > 0 else 0
        degraded_rate = degraded_count / count if count > 0 else 0

        total_error_rate += error_rate
        stage_count += 1

        # Determine overall status for this stage
        if error_rate > 0.1:
            status = "error"
        elif degraded_rate > 0.2:
            status = "degraded"
        else:
            status = "ok"

        # Get latest metadata
        latest = stage_events[0] if stage_events else None
        metadata = json.loads(latest.metadata_json) if latest and latest.metadata_json else {}

        workflow.append({
            "stage": s,
            "status": status,
            "latency_ms": round(p50, 2),
            "metadata": metadata,
            "metrics": {
                "count": count,
                "p50_latency_ms": round(p50, 2),
                "p90_latency_ms": round(p90, 2),
                "p99_latency_ms": round(p99, 2),
                "error_rate": round(error_rate, 4),
                "degraded_rate": round(degraded_rate, 4),
            }
        })

    # Determine safe_mode based on aggregate error rate
    avg_error_rate = total_error_rate / stage_count if stage_count > 0 else 0
    safe_mode = avg_error_rate > 0.05 or any(w["status"] == "error" for w in workflow)

    return {
        "fleet_id": fleet_id,
        "timestamp": datetime.utcnow().isoformat(),
        "time_range": time_range,
        "safe_mode": safe_mode,
        "workflow": workflow,
        "summary": {
            "total_events": len(events),
            "avg_error_rate": round(avg_error_rate, 4),
            "stages_with_errors": sum(1 for w in workflow if w["status"] == "error"),
            "stages_degraded": sum(1 for w in workflow if w["status"] == "degraded"),
        },
    }


# =============================================================================
# Edge Telemetry Query (User Auth)
# =============================================================================

@router.get("/telemetry/edge")
async def get_edge_telemetry(
    fleet_id: Optional[str] = None,
    device_id: Optional[str] = None,
    since_minutes: int = Query(default=60, le=1440),
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get latest gating stage events per device.

    Auth: Bearer token (get_current_user)

    Returns real telemetry data from edge agents, never simulated.
    Replaces simulated GET /edge/telemetry in edge_gating_endpoints.py.
    """
    since = datetime.utcnow() - timedelta(minutes=since_minutes)

    query = select(TelemetryStageEvent).where(
        TelemetryStageEvent.tenant_id == current_user.tenant_id,
        TelemetryStageEvent.ts >= since,
    )

    if fleet_id:
        fleet = session.get(Fleet, fleet_id)
        if not fleet or fleet.tenant_id != current_user.tenant_id:
            raise HTTPException(status_code=404, detail="Fleet not found")
        query = query.where(TelemetryStageEvent.fleet_id == fleet_id)

    if device_id:
        query = query.where(TelemetryStageEvent.device_id == device_id)

    total = session.exec(select(func.count()).select_from(query.subquery())).one()

    events = session.exec(
        query.order_by(TelemetryStageEvent.ts.desc()).offset(offset).limit(limit)
    ).all()

    return {
        "telemetry": [
            {
                "id": e.id,
                "device_id": e.device_id,
                "fleet_id": e.fleet_id,
                "stage": e.stage,
                "status": e.status,
                "latency_ms": e.latency_ms,
                "run_id": e.run_id,
                "metadata": json.loads(e.metadata_json) if e.metadata_json else {},
                "timestamp": e.ts.isoformat(),
            }
            for e in events
        ],
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        "since": since.isoformat(),
    }


# =============================================================================
# System Telemetry Query (User Auth)
# =============================================================================

@router.get("/telemetry/system")
async def get_system_telemetry(
    fleet_id: Optional[str] = None,
    device_id: Optional[str] = None,
    since_minutes: int = Query(default=60, le=1440),
    limit: int = Query(default=100, le=1000),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get system resource telemetry from edge devices.

    Auth: Bearer token (get_current_user)
    """
    since = datetime.utcnow() - timedelta(minutes=since_minutes)

    query = select(TelemetrySystemEvent).where(
        TelemetrySystemEvent.tenant_id == current_user.tenant_id,
        TelemetrySystemEvent.ts >= since,
    )

    if fleet_id:
        fleet = session.get(Fleet, fleet_id)
        if not fleet or fleet.tenant_id != current_user.tenant_id:
            raise HTTPException(status_code=404, detail="Fleet not found")
        query = query.where(TelemetrySystemEvent.fleet_id == fleet_id)

    if device_id:
        query = query.where(TelemetrySystemEvent.device_id == device_id)

    events = session.exec(
        query.order_by(TelemetrySystemEvent.ts.desc()).limit(limit)
    ).all()

    return {
        "system_telemetry": [
            {
                "device_id": e.device_id,
                "cpu_pct": e.cpu_pct,
                "mem_pct": e.mem_pct,
                "gpu_pct": e.gpu_pct,
                "temp_c": e.temp_c,
                "bandwidth_up_bps": e.bandwidth_up_bps,
                "bandwidth_down_bps": e.bandwidth_down_bps,
                "dropped_frames": e.dropped_frames,
                "queue_latency_ms": e.queue_latency_ms,
                "timestamp": e.ts.isoformat(),
            }
            for e in events
        ],
        "total": len(events),
        "since": since.isoformat(),
    }


# =============================================================================
# Model Behavior Telemetry Query (User Auth)
# =============================================================================

@router.get("/telemetry/model_behavior")
async def get_model_behavior_telemetry(
    fleet_id: Optional[str] = None,
    device_id: Optional[str] = None,
    model_version: Optional[str] = None,
    is_shadow: Optional[bool] = None,
    since_minutes: int = Query(default=60, le=1440),
    limit: int = Query(default=100, le=1000),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get model behavior telemetry for shadow/A-B analysis.

    Auth: Bearer token (get_current_user)
    """
    since = datetime.utcnow() - timedelta(minutes=since_minutes)

    query = select(TelemetryModelBehaviorEvent).where(
        TelemetryModelBehaviorEvent.tenant_id == current_user.tenant_id,
        TelemetryModelBehaviorEvent.ts >= since,
    )

    if fleet_id:
        fleet = session.get(Fleet, fleet_id)
        if not fleet or fleet.tenant_id != current_user.tenant_id:
            raise HTTPException(status_code=404, detail="Fleet not found")
        query = query.where(TelemetryModelBehaviorEvent.fleet_id == fleet_id)

    if device_id:
        query = query.where(TelemetryModelBehaviorEvent.device_id == device_id)

    if model_version:
        query = query.where(TelemetryModelBehaviorEvent.model_version == model_version)

    if is_shadow is not None:
        query = query.where(TelemetryModelBehaviorEvent.is_shadow == is_shadow)

    events = session.exec(
        query.order_by(TelemetryModelBehaviorEvent.ts.desc()).limit(limit)
    ).all()

    return {
        "model_behavior_telemetry": [
            {
                "device_id": e.device_id,
                "model_version": e.model_version,
                "adapter_id": e.adapter_id,
                "decision_hash": e.decision_hash,
                "refusal_rate": e.refusal_rate,
                "tool_call_failures": e.tool_call_failures,
                "policy_constraint_hits": e.policy_constraint_hits,
                "is_shadow": e.is_shadow,
                "timestamp": e.ts.isoformat(),
            }
            for e in events
        ],
        "total": len(events),
        "since": since.isoformat(),
    }


# =============================================================================
# Forensics Events Query (User Auth)
# =============================================================================

@router.get("/telemetry/forensics")
async def get_forensics_events(
    fleet_id: Optional[str] = None,
    device_id: Optional[str] = None,
    severity: Optional[str] = None,
    event_type: Optional[str] = None,
    since_minutes: int = Query(default=1440, le=10080),  # Default 24h, max 1 week
    limit: int = Query(default=100, le=1000),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get forensics events for incident investigation.

    Auth: Bearer token (get_current_user)

    Replaces mock forensics/incidents endpoint.
    """
    since = datetime.utcnow() - timedelta(minutes=since_minutes)

    query = select(ForensicsEvent).where(
        ForensicsEvent.tenant_id == current_user.tenant_id,
        ForensicsEvent.ts >= since,
    )

    if fleet_id:
        fleet = session.get(Fleet, fleet_id)
        if not fleet or fleet.tenant_id != current_user.tenant_id:
            raise HTTPException(status_code=404, detail="Fleet not found")
        query = query.where(ForensicsEvent.fleet_id == fleet_id)

    if device_id:
        query = query.where(ForensicsEvent.device_id == device_id)

    if severity:
        query = query.where(ForensicsEvent.severity == severity)

    if event_type:
        query = query.where(ForensicsEvent.event_type == event_type)

    events = session.exec(
        query.order_by(ForensicsEvent.ts.desc()).limit(limit)
    ).all()

    return {
        "forensics_events": [
            {
                "id": e.id,
                "device_id": e.device_id,
                "fleet_id": e.fleet_id,
                "severity": e.severity,
                "event_type": e.event_type,
                "evidence_ref": e.evidence_ref,
                "details": json.loads(e.details_json) if e.details_json else {},
                "timestamp": e.ts.isoformat(),
            }
            for e in events
        ],
        "total": len(events),
        "since": since.isoformat(),
    }


# =============================================================================
# Device Registry Query (User Auth)
# =============================================================================

@router.get("/telemetry/devices")
async def get_fleet_devices(
    fleet_id: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get registered devices with version information.

    Auth: Bearer token (get_current_user)
    """
    query = select(FleetDevice).where(
        FleetDevice.tenant_id == current_user.tenant_id,
    )

    if fleet_id:
        fleet = session.get(Fleet, fleet_id)
        if not fleet or fleet.tenant_id != current_user.tenant_id:
            raise HTTPException(status_code=404, detail="Fleet not found")
        query = query.where(FleetDevice.fleet_id == fleet_id)

    devices = session.exec(
        query.order_by(FleetDevice.last_seen_at.desc()).limit(limit)
    ).all()

    return {
        "devices": [
            {
                "device_id": d.device_id,
                "fleet_id": d.fleet_id,
                "agent_version": d.agent_version,
                "runtime_version": d.runtime_version,
                "ros_distro": d.ros_distro,
                "firmware_version": d.firmware_version,
                "sensor_manifest_hash": d.sensor_manifest_hash,
                "last_seen_at": d.last_seen_at.isoformat() if d.last_seen_at else None,
                "created_at": d.created_at.isoformat(),
            }
            for d in devices
        ],
        "total": len(devices),
    }
