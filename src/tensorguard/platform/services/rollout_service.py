"""
Rollout Service for Production Deployment Strategy

Responsibilities:
- Deterministic cohort assignment (hash-based, no randomness)
- Compatibility enforcement (version checks)
- Rollback trigger evaluation
- Deployment directive computation for agents

All cohort assignments are deterministic and reproducible.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

from sqlmodel import Session, select, func

from ..models.rollout_models import (
    DeploymentPlan,
    DeploymentAssignment,
    RollbackEvent,
    CompatibilityEvent,
    DeploymentStatus,
    DeploymentMode,
    RollbackTriggerType,
    AssignmentVariant,
)
from ..models.telemetry_models import (
    FleetDevice,
    TelemetryStageEvent,
    ForensicsEvent,
    StageStatus,
)
from ..models.core import AuditLog
from .audit import AuditService

logger = logging.getLogger(__name__)


# Bucket range for deterministic assignment (0-9999 = 10000 buckets)
BUCKET_COUNT = 10000


class RolloutService:
    """
    Service for managing deployment rollouts.

    All operations are deterministic and reproducible:
    - Cohort assignment uses stable hashing
    - Same device_id always maps to same bucket
    - Same bucket always gets same assignment for a given deployment
    """

    def __init__(self, session: Session):
        self.session = session

    # =========================================================================
    # Deterministic Cohort Assignment
    # =========================================================================

    @staticmethod
    def compute_bucket(device_id: str) -> int:
        """
        Compute deterministic bucket for a device.

        Uses SHA256 hash of device_id to map to bucket 0-9999.
        Same device_id always produces same bucket.
        """
        hash_bytes = hashlib.sha256(device_id.encode()).digest()
        # Use first 4 bytes as unsigned int, mod BUCKET_COUNT
        bucket = int.from_bytes(hash_bytes[:4], byteorder='big') % BUCKET_COUNT
        return bucket

    def assign_device(
        self,
        deployment: DeploymentPlan,
        device_id: str,
        force: bool = False
    ) -> Optional[DeploymentAssignment]:
        """
        Assign a device to a deployment based on deterministic cohort.

        Returns assignment if device is in the active cohort, None otherwise.
        """
        # Check if assignment already exists
        existing = self.session.exec(
            select(DeploymentAssignment).where(
                DeploymentAssignment.deployment_id == deployment.id,
                DeploymentAssignment.device_id == device_id
            )
        ).first()

        if existing and not force:
            return existing

        # Compute bucket
        bucket = self.compute_bucket(device_id)

        # Get stage configuration
        stages = deployment.get_stages()
        current_stage = deployment.current_stage

        # Determine if device is in active cohort
        if deployment.mode == DeploymentMode.CANARY.value:
            threshold = stages.get("canary_pct", 10) * (BUCKET_COUNT // 100)
            in_cohort = bucket < threshold

        elif deployment.mode == DeploymentMode.AB.value:
            # A/B split: even buckets = A, odd buckets = B
            in_cohort = True  # All devices get assigned
            variant = AssignmentVariant.A.value if bucket % 2 == 0 else AssignmentVariant.B.value

        elif deployment.mode == DeploymentMode.SHADOW.value:
            # Shadow: all devices get shadow assignment
            in_cohort = True

        elif deployment.mode == DeploymentMode.FULL.value:
            # Full: all devices get assignment
            in_cohort = True

        else:
            in_cohort = False

        if not in_cohort and not force:
            return None

        # Determine assignment details
        assigned_adapter_id = deployment.target_adapter_id
        is_shadow = deployment.mode == DeploymentMode.SHADOW.value
        assigned_variant = None

        if deployment.mode == DeploymentMode.AB.value:
            assigned_variant = AssignmentVariant.A.value if bucket % 2 == 0 else AssignmentVariant.B.value
            # B variant gets the new adapter, A keeps current
            if assigned_variant == AssignmentVariant.A.value:
                assigned_adapter_id = deployment.previous_adapter_id

        # Create or update assignment
        if existing:
            existing.assigned_adapter_id = assigned_adapter_id
            existing.assigned_variant = assigned_variant
            existing.is_shadow = is_shadow
            existing.bucket = bucket
            existing.updated_at = datetime.utcnow()
            return existing

        assignment = DeploymentAssignment(
            tenant_id=deployment.tenant_id,
            fleet_id=deployment.fleet_id,
            device_id=device_id,
            deployment_id=deployment.id,
            assigned_adapter_id=assigned_adapter_id,
            assigned_variant=assigned_variant,
            is_shadow=is_shadow,
            bucket=bucket,
        )
        self.session.add(assignment)
        return assignment

    def get_deployment_directive(
        self,
        device_id: str,
        fleet_id: str,
        tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Compute deployment directive for a device.

        Returns directive with adapter_id, model_version, variant, is_shadow
        if there's an active deployment, None otherwise.
        """
        # Find active deployment for fleet
        deployment = self.session.exec(
            select(DeploymentPlan).where(
                DeploymentPlan.fleet_id == fleet_id,
                DeploymentPlan.tenant_id == tenant_id,
                DeploymentPlan.status == DeploymentStatus.RUNNING.value
            )
        ).first()

        if not deployment:
            return None

        # Get or create assignment
        assignment = self.assign_device(deployment, device_id)

        if not assignment:
            return None

        return {
            "deployment_id": deployment.id,
            "adapter_id": assignment.assigned_adapter_id,
            "model_version": deployment.target_model_version,
            "variant": assignment.assigned_variant,
            "is_shadow": assignment.is_shadow,
            "effective_at": datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # Compatibility Enforcement
    # =========================================================================

    def check_compatibility(
        self,
        deployment: DeploymentPlan,
        device: FleetDevice
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Check if a device meets deployment compatibility requirements.

        Returns (is_compatible, list of check results).
        """
        compatibility = deployment.get_compatibility()
        results = []
        all_passed = True

        # Check agent version
        if "min_agent_version" in compatibility:
            required = compatibility["min_agent_version"]
            actual = device.agent_version
            passed = self._compare_versions(actual, required, ">=")
            results.append({
                "check_type": "agent_version",
                "required": required,
                "actual": actual,
                "passed": passed,
            })
            if not passed:
                all_passed = False
                self._record_compatibility_event(
                    deployment, device, "agent_version", required, actual, passed
                )

        # Check runtime version
        if "min_runtime_version" in compatibility:
            required = compatibility["min_runtime_version"]
            actual = device.runtime_version
            passed = self._compare_versions(actual, required, ">=")
            results.append({
                "check_type": "runtime_version",
                "required": required,
                "actual": actual,
                "passed": passed,
            })
            if not passed:
                all_passed = False
                self._record_compatibility_event(
                    deployment, device, "runtime_version", required, actual, passed
                )

        # Check firmware version
        if "min_firmware_version" in compatibility:
            required = compatibility["min_firmware_version"]
            actual = device.firmware_version
            passed = self._compare_versions(actual, required, ">=")
            results.append({
                "check_type": "firmware_version",
                "required": required,
                "actual": actual,
                "passed": passed,
            })
            if not passed:
                all_passed = False
                self._record_compatibility_event(
                    deployment, device, "firmware_version", required, actual, passed
                )

        # Check sensor manifest (allowlist)
        if "sensor_manifest_allowlist" in compatibility:
            allowlist = compatibility["sensor_manifest_allowlist"]
            actual = device.sensor_manifest_hash
            passed = actual in allowlist if actual and allowlist else True
            results.append({
                "check_type": "sensor_manifest",
                "required": "in allowlist",
                "actual": actual,
                "passed": passed,
            })
            if not passed:
                all_passed = False
                self._record_compatibility_event(
                    deployment, device, "sensor_manifest", str(allowlist), actual, passed
                )

        return all_passed, results

    def _compare_versions(
        self,
        actual: Optional[str],
        required: str,
        operator: str
    ) -> bool:
        """Compare semver versions."""
        if not actual:
            return False

        try:
            actual_parts = [int(x) for x in actual.split(".")[:3]]
            required_parts = [int(x) for x in required.split(".")[:3]]

            # Pad to 3 parts
            while len(actual_parts) < 3:
                actual_parts.append(0)
            while len(required_parts) < 3:
                required_parts.append(0)

            if operator == ">=":
                return actual_parts >= required_parts
            elif operator == ">":
                return actual_parts > required_parts
            elif operator == "==":
                return actual_parts == required_parts
            elif operator == "<=":
                return actual_parts <= required_parts
            elif operator == "<":
                return actual_parts < required_parts

        except (ValueError, AttributeError):
            return False

        return False

    def _record_compatibility_event(
        self,
        deployment: DeploymentPlan,
        device: FleetDevice,
        check_type: str,
        required: str,
        actual: Optional[str],
        passed: bool
    ) -> None:
        """Record a compatibility check event."""
        event = CompatibilityEvent(
            tenant_id=deployment.tenant_id,
            fleet_id=deployment.fleet_id,
            device_id=device.device_id,
            deployment_id=deployment.id,
            check_type=check_type,
            required_value=required,
            actual_value=actual,
            passed=passed,
        )
        self.session.add(event)

    # =========================================================================
    # Rollback Triggers
    # =========================================================================

    def evaluate_rollback_triggers(
        self,
        deployment: DeploymentPlan,
        window_minutes: int = 10
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Evaluate telemetry to determine if rollback should be triggered.

        Checks:
        - Stage error rate threshold
        - P99 latency threshold
        - Safety events threshold

        Returns (should_rollback, trigger_details).
        """
        guardrails = deployment.get_guardrails()
        since = datetime.utcnow() - timedelta(minutes=window_minutes)

        # Query telemetry
        events = self.session.exec(
            select(TelemetryStageEvent).where(
                TelemetryStageEvent.fleet_id == deployment.fleet_id,
                TelemetryStageEvent.ts >= since
            )
        ).all()

        if not events:
            return False, None

        # Calculate error rate
        total = len(events)
        errors = sum(1 for e in events if e.status == StageStatus.ERROR.value)
        error_rate = errors / total if total > 0 else 0

        error_threshold = guardrails.get("error_rate_threshold", 0.05)
        if error_rate > error_threshold:
            return True, {
                "trigger_type": RollbackTriggerType.TELEMETRY_THRESHOLD.value,
                "metric": "error_rate",
                "threshold": error_threshold,
                "actual": error_rate,
                "window_minutes": window_minutes,
            }

        # Calculate P99 latency
        latencies = sorted([e.latency_ms for e in events])
        if latencies:
            p99_idx = int(len(latencies) * 0.99)
            p99_latency = latencies[min(p99_idx, len(latencies) - 1)]

            latency_threshold = guardrails.get("p99_latency_threshold_ms", 500)
            if p99_latency > latency_threshold:
                return True, {
                    "trigger_type": RollbackTriggerType.TELEMETRY_THRESHOLD.value,
                    "metric": "p99_latency_ms",
                    "threshold": latency_threshold,
                    "actual": p99_latency,
                    "window_minutes": window_minutes,
                }

        # Check safety events
        safety_events = self.session.exec(
            select(func.count(ForensicsEvent.id)).where(
                ForensicsEvent.fleet_id == deployment.fleet_id,
                ForensicsEvent.ts >= since,
                ForensicsEvent.severity.in_(["critical", "high"])
            )
        ).one()

        safety_threshold = guardrails.get("safety_event_threshold", 3)
        if safety_events > safety_threshold:
            return True, {
                "trigger_type": RollbackTriggerType.SAFETY_EVENT.value,
                "metric": "safety_events",
                "threshold": safety_threshold,
                "actual": safety_events,
                "window_minutes": window_minutes,
            }

        return False, None

    def trigger_rollback(
        self,
        deployment: DeploymentPlan,
        trigger_type: str,
        trigger_details: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> RollbackEvent:
        """
        Execute a rollback for a deployment.

        - Marks deployment as rolled_back
        - Updates assignments to previous adapter
        - Records RollbackEvent
        - Creates AuditLog entry
        """
        # Update deployment status
        deployment.status = DeploymentStatus.ROLLED_BACK.value
        deployment.updated_at = datetime.utcnow()

        # Update all assignments to previous adapter
        assignments = self.session.exec(
            select(DeploymentAssignment).where(
                DeploymentAssignment.deployment_id == deployment.id
            )
        ).all()

        for assignment in assignments:
            assignment.assigned_adapter_id = deployment.previous_adapter_id
            assignment.is_shadow = False
            assignment.updated_at = datetime.utcnow()

        # Record rollback event
        rollback_event = RollbackEvent(
            tenant_id=deployment.tenant_id,
            fleet_id=deployment.fleet_id,
            deployment_id=deployment.id,
            trigger_type=trigger_type,
            trigger_details_json=json.dumps(trigger_details),
        )
        self.session.add(rollback_event)

        # Audit log
        AuditService.log(
            session=self.session,
            tenant_id=deployment.tenant_id,
            action="DEPLOYMENT_ROLLBACK",
            resource_id=deployment.id,
            resource_type="deployment",
            user_id=user_id,
            details={
                "trigger_type": trigger_type,
                **trigger_details
            },
            success=True
        )

        self.session.commit()

        logger.warning(
            f"Deployment {deployment.id} rolled back: {trigger_type} - {trigger_details}"
        )

        return rollback_event

    # =========================================================================
    # Deployment Lifecycle
    # =========================================================================

    def start_deployment(
        self,
        deployment: DeploymentPlan,
        user_id: Optional[str] = None
    ) -> None:
        """Start a deployment (draft -> running)."""
        if deployment.status != DeploymentStatus.DRAFT.value:
            raise ValueError(f"Cannot start deployment in status {deployment.status}")

        deployment.status = DeploymentStatus.RUNNING.value
        deployment.updated_at = datetime.utcnow()

        AuditService.log(
            session=self.session,
            tenant_id=deployment.tenant_id,
            action="DEPLOYMENT_START",
            resource_id=deployment.id,
            resource_type="deployment",
            user_id=user_id,
            details={"mode": deployment.mode, "target": deployment.target_adapter_id},
            success=True
        )

        self.session.commit()

    def promote_deployment(
        self,
        deployment: DeploymentPlan,
        user_id: Optional[str] = None
    ) -> None:
        """Promote deployment to next stage (canary -> cohort -> full)."""
        if deployment.status != DeploymentStatus.RUNNING.value:
            raise ValueError(f"Cannot promote deployment in status {deployment.status}")

        stages = deployment.get_stages()
        stage_order = ["canary", "cohort", "full"]

        current = deployment.current_stage
        if current >= len(stage_order) - 1:
            # Already at full, mark as completed
            deployment.status = DeploymentStatus.COMPLETED.value
        else:
            deployment.current_stage = current + 1

        deployment.updated_at = datetime.utcnow()

        AuditService.log(
            session=self.session,
            tenant_id=deployment.tenant_id,
            action="DEPLOYMENT_PROMOTE",
            resource_id=deployment.id,
            resource_type="deployment",
            user_id=user_id,
            details={"from_stage": current, "to_stage": deployment.current_stage},
            success=True
        )

        self.session.commit()

    def pause_deployment(
        self,
        deployment: DeploymentPlan,
        user_id: Optional[str] = None
    ) -> None:
        """Pause a running deployment."""
        if deployment.status != DeploymentStatus.RUNNING.value:
            raise ValueError(f"Cannot pause deployment in status {deployment.status}")

        deployment.status = DeploymentStatus.PAUSED.value
        deployment.updated_at = datetime.utcnow()

        AuditService.log(
            session=self.session,
            tenant_id=deployment.tenant_id,
            action="DEPLOYMENT_PAUSE",
            resource_id=deployment.id,
            resource_type="deployment",
            user_id=user_id,
            details={},
            success=True
        )

        self.session.commit()

    def resume_deployment(
        self,
        deployment: DeploymentPlan,
        user_id: Optional[str] = None
    ) -> None:
        """Resume a paused deployment."""
        if deployment.status != DeploymentStatus.PAUSED.value:
            raise ValueError(f"Cannot resume deployment in status {deployment.status}")

        deployment.status = DeploymentStatus.RUNNING.value
        deployment.updated_at = datetime.utcnow()

        AuditService.log(
            session=self.session,
            tenant_id=deployment.tenant_id,
            action="DEPLOYMENT_RESUME",
            resource_id=deployment.id,
            resource_type="deployment",
            user_id=user_id,
            details={},
            success=True
        )

        self.session.commit()

    def get_guardrail_metrics(
        self,
        deployment: DeploymentPlan,
        window_minutes: int = 10
    ) -> Dict[str, Any]:
        """
        Get current guardrail metrics snapshot for a deployment.

        Used for deployment status endpoints to show current health.
        """
        since = datetime.utcnow() - timedelta(minutes=window_minutes)

        # Query telemetry
        events = self.session.exec(
            select(TelemetryStageEvent).where(
                TelemetryStageEvent.fleet_id == deployment.fleet_id,
                TelemetryStageEvent.ts >= since
            )
        ).all()

        if not events:
            return {
                "error_rate": 0,
                "p50_latency_ms": 0,
                "p90_latency_ms": 0,
                "p99_latency_ms": 0,
                "total_events": 0,
                "safety_events": 0,
                "window_minutes": window_minutes,
            }

        total = len(events)
        errors = sum(1 for e in events if e.status == StageStatus.ERROR.value)
        latencies = sorted([e.latency_ms for e in events])

        p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0
        p90 = latencies[int(len(latencies) * 0.90)] if latencies else 0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0

        # Safety events count
        safety_events = self.session.exec(
            select(func.count(ForensicsEvent.id)).where(
                ForensicsEvent.fleet_id == deployment.fleet_id,
                ForensicsEvent.ts >= since,
                ForensicsEvent.severity.in_(["critical", "high"])
            )
        ).one()

        return {
            "error_rate": errors / total if total > 0 else 0,
            "p50_latency_ms": p50,
            "p90_latency_ms": p90,
            "p99_latency_ms": p99,
            "total_events": total,
            "error_events": errors,
            "safety_events": safety_events,
            "window_minutes": window_minutes,
        }
