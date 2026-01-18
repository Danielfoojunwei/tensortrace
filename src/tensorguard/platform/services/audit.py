import json
import logging
from typing import Optional, Any, Dict
from datetime import datetime
from sqlmodel import Session
from fastapi import Request

from ..models.core import AuditLog

logger = logging.getLogger(__name__)

class AuditService:
    """
    Centralized service for recording security and operational events.
    Supports SOC 2, HIPAA, and GDPR traceability requirements.
    """
    
    @staticmethod
    def log(
        session: Session,
        tenant_id: str,
        action: str,
        resource_id: str,
        resource_type: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None,
        success: bool = True
    ):
        """Record an event to the audit ledger."""
        try:
            ip_address = None
            if request:
                ip_address = request.client.host
                
            entry = AuditLog(
                tenant_id=tenant_id,
                user_id=user_id,
                action=action,
                resource_id=resource_id,
                resource_type=resource_type,
                details=json.dumps(details or {}),
                ip_address=ip_address,
                success=success
            )
            session.add(entry)
            session.commit()
            
            logger.info(f"Audit: {action} on {resource_type}/{resource_id} (Success: {success})")
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            # Do not raise - we don't want audit failures to break the primary transaction,
            # though in some high-security modes we might want the opposite.
            session.rollback()

    @staticmethod
    def log_security_event(session: Session, tenant_id: str, action: str, details: Dict[str, Any]):
        """Helper for security-specific events (login, unauthorized access)."""
        AuditService.log(
            session=session,
            tenant_id=tenant_id,
            action=action,
            resource_id="system",
            resource_type="security",
            details=details
        )
