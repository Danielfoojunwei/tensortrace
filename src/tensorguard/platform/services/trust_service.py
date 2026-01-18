from sqlmodel import Session, select
from typing import Dict, Any, List
from ..models.identity_models import IdentityCertificate, IdentityRenewalJob, RenewalJobStatus
from ..models.core import Fleet, AuditLog
from ...identity.inventory import InventoryService
from ...core.keys import vault, KeyScope
import hashlib
from datetime import datetime, UTC

class TrustService:
    """
    Unified Security Engine for TensorGuardFlow.
    Aggregates Identity (Transport), N2HE (Privacy), and PQC (Integrity).
    """
    def __init__(self, session: Session):
        self.session = session
        self.inventory = InventoryService(session)

    def calculate_fleet_trust(self, fleet_id: str) -> Dict[str, Any]:
        """Calculate the Triple-Trust Score for a fleet."""
        # 1. Transport Layer (Identity Guard)
        certs = self.inventory.list_certificates(fleet_id=fleet_id, tenant_id=None) # Scoped by auth in caller
        
        id_score = 100
        violations = []
        
        if not certs:
            id_score = 0
            violations.append("NO_IDENTITY_CERT")
        else:
            current_cert = certs[0] # Most recent
            if current_cert.days_to_expiry < 7:
                id_score -= 50
                violations.append("EXPIRY_CRITICAL")
            elif current_cert.days_to_expiry < 30:
                id_score -= 20
                violations.append("EXPIRY_WARNING")
                
            if current_cert.has_eku_conflict:
                id_score -= 30
                violations.append("CHROME_2026_INCOMPATIBLE")

        # 2. Data Layer (N2HE Privacy)
        # Check if encryption keys are active in the Unified Key Fabric
        n2he_keys = vault.list_keys(scope=KeyScope.AGGREGATION)
        fleet_n2he_keys = [k for k in n2he_keys if k.get("params", {}).get("fleet_id") == fleet_id]
        
        privacy_score = 100
        if not fleet_n2he_keys:
            privacy_score = 50 # Unencrypted aggregation risk
            violations.append("N2HE_MISSING")
        else:
            # Check rotation (e.g., > 90 days is risky)
            last_key = max(fleet_n2he_keys, key=lambda x: x["created_at"])
            dt = datetime.fromisoformat(last_key["created_at"])
            age_days = (datetime.now(UTC) - dt).days
            if age_days > 90:
                privacy_score -= 20
                violations.append("N2HE_ROTATION_DUE")

        # 3. Audit Layer (PQC Integrity)
        # Verify the presence of PQC signatures in most recent logs
        recent_logs = self.session.exec(
            select(AuditLog).where(AuditLog.resource_id == fleet_id).order_by(AuditLog.timestamp.desc()).limit(10)
        ).all()
        
        integrity_score = 100
        if recent_logs:
            signed_count = sum(1 for log in recent_logs if log.pqc_signature)
            signed_ratio = signed_count / len(recent_logs)
            integrity_score = int(signed_ratio * 100)
            if signed_ratio < 0.5:
                violations.append("PQC_SIGNING_OFFLINE")
        else:
            integrity_score = 100 # Default if no activity
        
        aggregate_score = (id_score * 0.4) + (privacy_score * 0.4) + (integrity_score * 0.2)
        
        return {
            "fleet_id": fleet_id,
            "aggregate_score": round(aggregate_score, 1),
            "layers": {
                "transport": {"status": "ok" if id_score > 70 else "fail", "score": id_score},
                "privacy": {"status": "ok" if privacy_score > 70 else "fail", "score": privacy_score},
                "integrity": {"status": "ok" if integrity_score > 70 else "fail", "score": integrity_score}
            },
            "system_flags": violations,
            "quantum_ready": integrity_score >= 90
        }

    def get_global_posture(self, tenant_id: str) -> Dict[str, Any]:
        """Aggregate posture for all fleets in a tenant."""
        fleets = self.session.exec(select(Fleet).where(Fleet.tenant_id == tenant_id)).all()
        scores = [self.calculate_fleet_trust(f.id)["aggregate_score"] for f in fleets]
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "tenant_id": tenant_id,
            "compliance_health": round(avg_score, 1),
            "at_risk_fleets": sum(1 for s in scores if s < 70),
            "threat_environment": "STABLE" if avg_score > 80 else "ELEVATED"
        }
