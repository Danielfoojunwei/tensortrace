"""
Forensics Service

Provides forensics-grade logging with cryptographic signatures for audit trails.
Supports PQC (Post-Quantum Cryptography) signatures for tamper-evident records.

Used for adapter swaps, rollbacks, and other security-critical events.
"""

import json
import logging
from functools import lru_cache
from datetime import datetime
from typing import Optional, Dict, Any
from sqlmodel import Session

from ..models.core import AuditLog
from ..models.telemetry_models import ForensicsEvent
from ...crypto.pqc.dilithium import Dilithium3
from ...utils.production_gates import require_env

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _load_pqc_keys() -> Dict[str, bytes]:
    private_key_hex = require_env(
        "TG_FORENSICS_PQC_PRIVATE_KEY",
        remediation=(
            "Provide the Dilithium-3 private key hex in TG_FORENSICS_PQC_PRIVATE_KEY."
        ),
    )
    public_key_hex = require_env(
        "TG_FORENSICS_PQC_PUBLIC_KEY",
        remediation=(
            "Provide the Dilithium-3 public key hex in TG_FORENSICS_PQC_PUBLIC_KEY."
        ),
    )
    if not private_key_hex or not public_key_hex:
        raise RuntimeError("Forensics PQC keys must be configured.")
    try:
        return {
            "private": bytes.fromhex(private_key_hex),
            "public": bytes.fromhex(public_key_hex),
        }
    except ValueError as exc:
        raise ValueError("Invalid hex encoding for forensics PQC keys.") from exc


class ForensicsService:
    """
    Service for creating and verifying forensics-grade audit records.

    Features:
    - PQC signature generation for tamper evidence
    - Structured forensics event storage
    - Audit trail integration
    """

    def __init__(self, session: Session):
        self.session = session

    def sign_and_log(
        self,
        tenant_id: str,
        fleet_id: str,
        device_id: str,
        event_type: str,
        deployment_id: str,
        adapter_id: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> ForensicsEvent:
        """
        Create a forensics-grade event with PQC signature.

        Args:
            tenant_id: Tenant ID
            fleet_id: Fleet ID
            device_id: Device ID that initiated the event
            event_type: Type of event (adapter_swap, rollback, etc.)
            deployment_id: ID of the deployment
            adapter_id: ID of the adapter involved
            details: Additional event details
            user_id: User ID if event was user-initiated

        Returns:
            ForensicsEvent record with PQC signature
        """
        timestamp = datetime.utcnow()
        details_dict = details or {}

        # Compute PQC signature
        pqc_signature = self._compute_pqc_signature(
            deployment_id=deployment_id,
            adapter_id=adapter_id,
            timestamp=timestamp
        )

        # Create forensics event
        event = ForensicsEvent(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            device_id=device_id,
            event_type=event_type,
            deployment_id=deployment_id,
            adapter_id=adapter_id,
            details_json=json.dumps(details_dict),
            pqc_signature=pqc_signature,
            ts=timestamp
        )

        self.session.add(event)

        # Also log to audit trail
        self._log_to_audit(
            tenant_id=tenant_id,
            event_type=event_type,
            deployment_id=deployment_id,
            adapter_id=adapter_id,
            device_id=device_id,
            user_id=user_id,
            pqc_signature=pqc_signature
        )

        self.session.commit()
        self.session.refresh(event)

        logger.info(
            f"Forensics event logged: type={event_type}, "
            f"deployment={deployment_id}, device={device_id}, "
            f"signature={pqc_signature[:16]}..."
        )

        return event

    def verify_signature(self, event: ForensicsEvent) -> bool:
        """
        Verify the PQC signature of a forensics event.

        Args:
            event: ForensicsEvent to verify

        Returns:
            True if signature is valid, False otherwise
        """
        if not event.pqc_signature:
            logger.warning(f"Forensics event missing signature: event_id={event.id}")
            return False

        prefix = "pqc-dilithium3:"
        if not event.pqc_signature.startswith(prefix):
            logger.warning(f"Unsupported signature format for event_id={event.id}")
            return False

        signature_hex = event.pqc_signature[len(prefix):]
        try:
            signature_bytes = bytes.fromhex(signature_hex)
        except ValueError:
            logger.warning(f"Invalid signature encoding for event_id={event.id}")
            return False

        keys = _load_pqc_keys()
        pqc = Dilithium3()
        message = f"{event.deployment_id}:{event.adapter_id}:{event.ts.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}"

        is_valid = pqc.verify(keys["public"], message.encode(), signature_bytes)
        if not is_valid:
            logger.warning(
                f"Forensics event signature verification failed: event_id={event.id}"
            )
        return is_valid

    def log_adapter_swap(
        self,
        tenant_id: str,
        fleet_id: str,
        device_id: str,
        deployment_id: str,
        previous_adapter_id: str,
        new_adapter_id: str,
        is_rollback: bool = False,
        user_id: Optional[str] = None
    ) -> ForensicsEvent:
        """
        Log an adapter swap event with forensics-grade signature.

        Args:
            tenant_id: Tenant ID
            fleet_id: Fleet ID
            device_id: Device ID
            deployment_id: Deployment ID
            previous_adapter_id: ID of the adapter being replaced
            new_adapter_id: ID of the new adapter
            is_rollback: Whether this is a rollback operation
            user_id: User ID if manually triggered

        Returns:
            ForensicsEvent record
        """
        event_type = "rollback" if is_rollback else "adapter_swap"

        return self.sign_and_log(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            device_id=device_id,
            event_type=event_type,
            deployment_id=deployment_id,
            adapter_id=new_adapter_id,
            details={
                "previous_adapter_id": previous_adapter_id,
                "new_adapter_id": new_adapter_id,
                "is_rollback": is_rollback,
            },
            user_id=user_id
        )

    def log_deployment_event(
        self,
        tenant_id: str,
        fleet_id: str,
        deployment_id: str,
        event_type: str,
        adapter_id: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> ForensicsEvent:
        """
        Log a deployment lifecycle event.

        Args:
            tenant_id: Tenant ID
            fleet_id: Fleet ID
            deployment_id: Deployment ID
            event_type: Event type (deployment_start, deployment_promote, etc.)
            adapter_id: Target adapter ID
            details: Additional details
            user_id: User ID who triggered the event

        Returns:
            ForensicsEvent record
        """
        return self.sign_and_log(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            device_id="platform",  # Platform-initiated events
            event_type=event_type,
            deployment_id=deployment_id,
            adapter_id=adapter_id,
            details=details,
            user_id=user_id
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _compute_pqc_signature(
        self,
        deployment_id: str,
        adapter_id: str,
        timestamp: datetime
    ) -> str:
        """
        Compute PQC signature for forensics event.

        Signature format: pqc-dilithium3:<hex-signature>
        """
        # Canonical timestamp format
        ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        # Message to sign: deployment_id + adapter_id + timestamp
        message = f"{deployment_id}:{adapter_id}:{ts_str}"
        keys = _load_pqc_keys()
        pqc = Dilithium3()
        signature_bytes = pqc.sign(keys["private"], message.encode())
        return f"pqc-dilithium3:{signature_bytes.hex()}"

    def _log_to_audit(
        self,
        tenant_id: str,
        event_type: str,
        deployment_id: str,
        adapter_id: str,
        device_id: str,
        user_id: Optional[str],
        pqc_signature: str
    ):
        """Log forensics event to main audit trail."""
        try:
            entry = AuditLog(
                tenant_id=tenant_id,
                user_id=user_id,
                action=f"FORENSICS_{event_type.upper()}",
                resource_id=deployment_id,
                resource_type="deployment",
                details=json.dumps({
                    "adapter_id": adapter_id,
                    "device_id": device_id,
                    "pqc_signature": pqc_signature,
                }),
                success=True
            )
            self.session.add(entry)

        except Exception as e:
            logger.error(f"Failed to log forensics event to audit: {e}")


def compute_pqc_signature_standalone(
    deployment_id: str,
    adapter_id: str,
    timestamp: datetime
) -> str:
    """
    Standalone function to compute PQC signature.

    Can be used by edge agents without creating a service instance.
    """
    ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    message = f"{deployment_id}:{adapter_id}:{ts_str}"

    keys = _load_pqc_keys()
    pqc = Dilithium3()
    signature_bytes = pqc.sign(keys["private"], message.encode())
    return f"pqc-dilithium3:{signature_bytes.hex()}"
