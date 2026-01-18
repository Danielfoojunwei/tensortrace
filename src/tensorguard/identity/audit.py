"""
Audit Service - Tamper-Evident Logging

Provides hash-chained audit entries for all identity operations.
Each entry includes prev_hash + entry_hash for integrity verification.
"""

import hashlib
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlmodel import Session, select
import logging

from ..platform.models.identity_models import (
    IdentityAuditLog,
    AuditAction,
)

logger = logging.getLogger(__name__)


class AuditService:
    """
    Tamper-evident audit logging for identity operations.
    
    Features:
    - Hash-chained entries (prev_hash + entry_hash)
    - Monotonic sequence numbers per tenant
    - Payload hashing for integrity
    - Evidence URI linking
    """
    
    GENESIS_HASH = "0" * 64  # Genesis block hash
    
    def __init__(self, session: Session):
        self.session = session
    
    def log(
        self,
        tenant_id: str,
        action: AuditAction,
        actor_type: str,
        actor_id: str,
        payload: Optional[Dict[str, Any]] = None,
        fleet_id: Optional[str] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        actor_ip: Optional[str] = None,
        evidence_uri: Optional[str] = None,
        action_detail: Optional[str] = None,
    ) -> IdentityAuditLog:
        """
        Create a new audit log entry.
        
        Automatically:
        - Assigns next sequence number
        - Computes payload hash
        - Links to previous entry via prev_hash
        - Computes entry_hash for tamper detection
        """
        # Get the previous entry for this tenant
        prev_entry = self._get_last_entry(tenant_id)
        
        if prev_entry:
            prev_hash = prev_entry.entry_hash
            sequence_number = prev_entry.sequence_number + 1
        else:
            prev_hash = self.GENESIS_HASH
            sequence_number = 1
        
        # Compute payload hash
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':')) if payload else "{}"
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
        
        # Compute entry hash
        timestamp = datetime.utcnow()
        entry_hash = self._compute_entry_hash(
            prev_hash=prev_hash,
            action=action.value,
            payload_hash=payload_hash,
            timestamp=timestamp
        )
        
        # PQC Quantum Signing (Simulated Dilithium-3)
        # In a real impl, this would call src/tensorguard/crypto/sig.py
        pqc_signature = hashlib.sha3_512(f"dilithium3:{entry_hash}".encode()).hexdigest()
        
        # Create entry
        entry = IdentityAuditLog(
            sequence_number=sequence_number,
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            actor_type=actor_type,
            actor_id=actor_id,
            actor_ip=actor_ip,
            action=action,
            action_detail=action_detail,
            target_type=target_type,
            target_id=target_id,
            payload_json=payload_json,
            payload_hash=payload_hash,
            evidence_uri=evidence_uri,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            pqc_signature=pqc_signature,
            timestamp=timestamp,
        )
        
        self.session.add(entry)
        self.session.commit()
        self.session.refresh(entry)
        
        logger.info(f"Audit: {action.value} by {actor_type}:{actor_id} on {target_type}:{target_id}")
        return entry
    
    def _get_last_entry(self, tenant_id: str) -> Optional[IdentityAuditLog]:
        """Get the most recent audit entry for a tenant."""
        statement = (
            select(IdentityAuditLog)
            .where(IdentityAuditLog.tenant_id == tenant_id)
            .order_by(IdentityAuditLog.sequence_number.desc())
            .limit(1)
        )
        return self.session.exec(statement).first()
    
    def _compute_entry_hash(
        self,
        prev_hash: str,
        action: str,
        payload_hash: str,
        timestamp: datetime
    ) -> str:
        """Compute the tamper-evident hash for an entry."""
        data = f"{prev_hash}:{action}:{payload_hash}:{timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_chain(self, tenant_id: str) -> Dict[str, Any]:
        """
        Verify the integrity of the audit chain for a tenant.
        
        Returns:
            Dict with verification results:
            - is_valid: bool
            - total_entries: int
            - first_invalid_sequence: Optional[int]
            - error_message: Optional[str]
        """
        entries = self._get_all_entries(tenant_id)
        
        if not entries:
            return {
                "is_valid": True,
                "total_entries": 0,
                "first_invalid_sequence": None,
                "error_message": None,
            }
        
        # Verify chain integrity
        expected_prev_hash = self.GENESIS_HASH
        
        for i, entry in enumerate(entries):
            # Check sequence
            if entry.sequence_number != i + 1:
                return {
                    "is_valid": False,
                    "total_entries": len(entries),
                    "first_invalid_sequence": entry.sequence_number,
                    "error_message": f"Sequence gap: expected {i+1}, got {entry.sequence_number}",
                }
            
            # Check prev_hash
            if entry.prev_hash != expected_prev_hash:
                return {
                    "is_valid": False,
                    "total_entries": len(entries),
                    "first_invalid_sequence": entry.sequence_number,
                    "error_message": f"Hash chain broken at sequence {entry.sequence_number}",
                }
            
            # Verify entry_hash
            computed_hash = self._compute_entry_hash(
                entry.prev_hash,
                entry.action.value,
                entry.payload_hash,
                entry.timestamp
            )
            if entry.entry_hash != computed_hash:
                return {
                    "is_valid": False,
                    "total_entries": len(entries),
                    "first_invalid_sequence": entry.sequence_number,
                    "error_message": f"Entry hash mismatch at sequence {entry.sequence_number}",
                }
            
            # Verify payload hash
            payload_hash = hashlib.sha256(entry.payload_json.encode()).hexdigest()
            if entry.payload_hash != payload_hash:
                return {
                    "is_valid": False,
                    "total_entries": len(entries),
                    "first_invalid_sequence": entry.sequence_number,
                    "error_message": f"Payload hash mismatch at sequence {entry.sequence_number}",
                }
            
            expected_prev_hash = entry.entry_hash
        
        return {
            "is_valid": True,
            "total_entries": len(entries),
            "first_invalid_sequence": None,
            "error_message": None,
        }
    
    def _get_all_entries(self, tenant_id: str) -> List[IdentityAuditLog]:
        """Get all audit entries for a tenant, ordered by sequence."""
        statement = (
            select(IdentityAuditLog)
            .where(IdentityAuditLog.tenant_id == tenant_id)
            .order_by(IdentityAuditLog.sequence_number.asc())
        )
        return list(self.session.exec(statement).all())
    
    def get_entries(
        self,
        tenant_id: str,
        fleet_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[IdentityAuditLog]:
        """Query audit entries with filters."""
        statement = select(IdentityAuditLog).where(
            IdentityAuditLog.tenant_id == tenant_id
        )
        
        if fleet_id:
            statement = statement.where(IdentityAuditLog.fleet_id == fleet_id)
        if action:
            statement = statement.where(IdentityAuditLog.action == action)
        if target_type:
            statement = statement.where(IdentityAuditLog.target_type == target_type)
        if target_id:
            statement = statement.where(IdentityAuditLog.target_id == target_id)
        if since:
            statement = statement.where(IdentityAuditLog.timestamp >= since)
        
        statement = (
            statement
            .order_by(IdentityAuditLog.timestamp.desc())
            .offset(offset)
            .limit(limit)
        )
        
        return list(self.session.exec(statement).all())
