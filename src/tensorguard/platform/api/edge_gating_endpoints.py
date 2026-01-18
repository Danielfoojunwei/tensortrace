"""
Tier 1: Task-Aware Edge Gating API - Production Hardened

Controls the local LoRA adapter gating and telemetry stream on edge nodes.
Uses database-backed state for production reliability.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from ..database import get_session
from ..auth import get_current_user
from ..models.core import User
from ..models.settings_models import (
    EdgeNode,
    EdgeNodeStatus,
    TelemetrySample,
    GatingDecisionLog,
)
from ...utils.production_gates import is_production

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class EdgeNodeCreate(BaseModel):
    """Request to register a new edge node."""

    node_id: str
    fleet_id: Optional[str] = None
    gating_enabled: bool = True
    local_threshold: float = 0.15
    task_whitelist: List[str] = []


class EdgeNodeConfig(BaseModel):
    """Request to update edge node configuration."""

    node_id: str
    gating_enabled: bool
    local_threshold: float
    task_whitelist: List[str]


class TelemetrySubmit(BaseModel):
    """Telemetry data submitted by edge agent."""

    node_id: str
    task: str
    relevance_score: float
    threshold: float
    decision: str  # PASS, BLOCK
    latency_ms: Optional[float] = None


class HeartbeatRequest(BaseModel):
    """Heartbeat from edge node."""

    node_id: str
    status: str = "online"
    ip_address: Optional[str] = None


# ============================================================================
# Edge Node Management
# ============================================================================


@router.get("/edge/nodes")
async def list_edge_nodes(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
    status: Optional[str] = None,
    fleet_id: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
):
    """
    List all registered edge nodes for the current tenant.

    Returns real database-backed node state.
    """
    query = select(EdgeNode).where(EdgeNode.tenant_id == current_user.tenant_id)

    if status:
        query = query.where(EdgeNode.status == status)
    if fleet_id:
        query = query.where(EdgeNode.fleet_id == fleet_id)

    query = query.order_by(EdgeNode.created_at.desc()).limit(limit)
    nodes = session.exec(query).all()

    return {
        "nodes": [
            {
                "id": n.id,
                "node_id": n.node_id,
                "fleet_id": n.fleet_id,
                "gating_enabled": n.gating_enabled,
                "local_threshold": n.local_threshold,
                "task_whitelist": json.loads(n.task_whitelist),
                "status": n.status,
                "last_heartbeat": n.last_heartbeat.isoformat() if n.last_heartbeat else None,
            }
            for n in nodes
        ],
        "total": len(nodes),
    }


@router.post("/edge/nodes")
async def register_edge_node(
    req: EdgeNodeCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Register a new edge node."""
    # Check for existing node
    existing = session.exec(
        select(EdgeNode).where(EdgeNode.node_id == req.node_id)
    ).first()

    if existing:
        raise HTTPException(400, f"Edge node already registered: {req.node_id}")

    node = EdgeNode(
        node_id=req.node_id,
        tenant_id=current_user.tenant_id,
        fleet_id=req.fleet_id,
        gating_enabled=req.gating_enabled,
        local_threshold=req.local_threshold,
        task_whitelist=json.dumps(req.task_whitelist),
        status=EdgeNodeStatus.OFFLINE.value,
    )

    session.add(node)
    session.commit()
    session.refresh(node)

    return {
        "id": node.id,
        "node_id": node.node_id,
        "status": "registered",
        "message": "Edge node registered successfully",
    }


@router.post("/edge/config")
async def update_edge_config(
    req: EdgeNodeConfig,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Update gating configuration for a specific edge node."""
    node = session.exec(
        select(EdgeNode)
        .where(EdgeNode.node_id == req.node_id)
        .where(EdgeNode.tenant_id == current_user.tenant_id)
    ).first()

    if not node:
        raise HTTPException(404, f"Edge node not found: {req.node_id}")

    node.gating_enabled = req.gating_enabled
    node.local_threshold = req.local_threshold
    node.task_whitelist = json.dumps(req.task_whitelist)
    node.updated_at = datetime.utcnow()

    session.add(node)
    session.commit()

    return {
        "status": "updated",
        "node_id": req.node_id,
        "config": {
            "gating_enabled": node.gating_enabled,
            "local_threshold": node.local_threshold,
            "task_whitelist": json.loads(node.task_whitelist),
        },
    }


@router.delete("/edge/nodes/{node_id}")
async def delete_edge_node(
    node_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Unregister an edge node."""
    node = session.exec(
        select(EdgeNode)
        .where(EdgeNode.node_id == node_id)
        .where(EdgeNode.tenant_id == current_user.tenant_id)
    ).first()

    if not node:
        raise HTTPException(404, f"Edge node not found: {node_id}")

    session.delete(node)
    session.commit()

    return {"status": "deleted", "node_id": node_id}


# ============================================================================
# Telemetry Submission (from Edge Agents)
# ============================================================================


@router.post("/edge/telemetry")
async def submit_telemetry(
    samples: List[TelemetrySubmit],
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Submit telemetry samples from edge agents.

    This is the production endpoint for real telemetry data.
    Edge agents POST their gating decisions here.
    """
    results = []

    for sample in samples:
        # Verify node exists and belongs to tenant
        node = session.exec(
            select(EdgeNode)
            .where(EdgeNode.node_id == sample.node_id)
            .where(EdgeNode.tenant_id == current_user.tenant_id)
        ).first()

        if not node:
            results.append(
                {"node_id": sample.node_id, "status": "error", "message": "Node not found"}
            )
            continue

        # Store telemetry sample
        telemetry = TelemetrySample(
            node_id=node.id,
            tenant_id=current_user.tenant_id,
            task=sample.task,
            relevance_score=sample.relevance_score,
            threshold=sample.threshold,
            decision=sample.decision,
            latency_ms=sample.latency_ms,
        )

        session.add(telemetry)
        results.append({"node_id": sample.node_id, "status": "accepted"})

    session.commit()

    return {"submitted": len(samples), "results": results}


@router.get("/edge/telemetry")
async def get_edge_telemetry(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
    node_id: Optional[str] = None,
    since_minutes: int = Query(default=60, le=1440),
    limit: int = Query(default=100, le=1000),
):
    """
    Get real telemetry data from edge nodes.

    Returns actual telemetry samples POSTed by edge agents.
    In production, this returns real data or empty results - never simulated data.
    """
    since = datetime.utcnow() - timedelta(minutes=since_minutes)

    query = select(TelemetrySample).where(
        TelemetrySample.tenant_id == current_user.tenant_id,
        TelemetrySample.timestamp >= since,
    )

    if node_id:
        # Look up internal node ID
        node = session.exec(
            select(EdgeNode)
            .where(EdgeNode.node_id == node_id)
            .where(EdgeNode.tenant_id == current_user.tenant_id)
        ).first()

        if not node:
            raise HTTPException(404, f"Edge node not found: {node_id}")

        query = query.where(TelemetrySample.node_id == node.id)

    query = query.order_by(TelemetrySample.timestamp.desc()).limit(limit)
    samples = session.exec(query).all()

    if not samples:
        # Return empty results with explanation - never simulate data
        return {
            "telemetry": [],
            "total": 0,
            "message": (
                "No telemetry data available. "
                "Ensure edge agents are configured to POST to /edge/telemetry."
            ),
            "since": since.isoformat(),
        }

    # Map internal node IDs back to external IDs
    node_map = {}
    node_ids = list(set(s.node_id for s in samples))
    nodes = session.exec(select(EdgeNode).where(EdgeNode.id.in_(node_ids))).all()
    for n in nodes:
        node_map[n.id] = n.node_id

    return {
        "telemetry": [
            {
                "node_id": node_map.get(s.node_id, s.node_id),
                "timestamp": s.timestamp.isoformat(),
                "task": s.task,
                "relevance_score": s.relevance_score,
                "threshold": s.threshold,
                "decision": s.decision,
                "latency_ms": s.latency_ms,
            }
            for s in samples
        ],
        "total": len(samples),
        "since": since.isoformat(),
    }


# ============================================================================
# Node Heartbeat
# ============================================================================


@router.post("/edge/heartbeat")
async def edge_heartbeat(
    req: HeartbeatRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Receive heartbeat from edge node.

    Updates node status and last_seen timestamp.
    """
    node = session.exec(
        select(EdgeNode)
        .where(EdgeNode.node_id == req.node_id)
        .where(EdgeNode.tenant_id == current_user.tenant_id)
    ).first()

    if not node:
        raise HTTPException(404, f"Edge node not found: {req.node_id}")

    node.status = req.status
    node.last_heartbeat = datetime.utcnow()
    if req.ip_address:
        node.last_ip_address = req.ip_address
    node.updated_at = datetime.utcnow()

    session.add(node)
    session.commit()

    return {
        "node_id": req.node_id,
        "status": "acknowledged",
        "server_time": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Analytics
# ============================================================================


@router.get("/edge/analytics/{node_id}")
async def get_node_analytics(
    node_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
    hours: int = Query(default=24, le=168),
):
    """
    Get gating analytics for a specific node.

    Returns aggregated decision statistics.
    """
    node = session.exec(
        select(EdgeNode)
        .where(EdgeNode.node_id == node_id)
        .where(EdgeNode.tenant_id == current_user.tenant_id)
    ).first()

    if not node:
        raise HTTPException(404, f"Edge node not found: {node_id}")

    since = datetime.utcnow() - timedelta(hours=hours)

    # Query telemetry samples
    samples = session.exec(
        select(TelemetrySample).where(
            TelemetrySample.node_id == node.id,
            TelemetrySample.timestamp >= since,
        )
    ).all()

    if not samples:
        return {
            "node_id": node_id,
            "period_hours": hours,
            "total_decisions": 0,
            "pass_count": 0,
            "block_count": 0,
            "pass_rate": 0.0,
            "avg_relevance_score": 0.0,
            "avg_latency_ms": 0.0,
            "message": "No telemetry data for this period",
        }

    total = len(samples)
    pass_count = sum(1 for s in samples if s.decision == "PASS")
    block_count = sum(1 for s in samples if s.decision == "BLOCK")
    avg_score = sum(s.relevance_score for s in samples) / total
    latencies = [s.latency_ms for s in samples if s.latency_ms is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "node_id": node_id,
        "period_hours": hours,
        "total_decisions": total,
        "pass_count": pass_count,
        "block_count": block_count,
        "pass_rate": round(pass_count / total * 100, 2),
        "avg_relevance_score": round(avg_score, 4),
        "avg_latency_ms": round(avg_latency, 2),
    }
