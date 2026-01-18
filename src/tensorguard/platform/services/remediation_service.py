from sqlmodel import Session, select
from typing import Dict, Any, List
from ..core.keys import vault, KeyScope
from ..core.crypto import N2HEParams, N2HEContext
from .models.identity_models import IdentityEndpoint, IdentityPolicy
from .services.inventory_service import InventoryService
from ..identity.scheduler import RenewalScheduler
import logging

logger = logging.getLogger(__name__)

class RemediationService:
    """
    Automated security enforcement and remediation.
    Closes the loop between Trust Scoring and Active Security.
    """
    def __init__(self, session: Session):
        self.session = session
        self.inventory = InventoryService(session)
        self.scheduler = RenewalScheduler(session)

    def rotate_n2he_key(self, fleet_id: str) -> str:
        """
        Force rotation of privacy keys for a fleet.
        Bound to Identity renewal for cross-layer trust.
        """
        logger.info(f"Remediation: Rotating N2HE keys for fleet {fleet_id}")
        
        # 1. Generate new key
        params = N2HEParams() # Standard fleet params
        ctx = N2HEContext(params)
        ctx.generate_keys()
        
        # 2. Save to vault with fleet metadata
        key_name = f"fleet_{fleet_id}_n2he_key"
        ctx.save_key(key_name)
        
        # 3. Update vault metadata to bind to fleet
        # Load back to get the file path/meta for direct manipulation if needed
        # But save_key already did it.
        
        return key_name

    def fix_trust_violations(self, fleet_id: str):
        """Analyze fleet trust and apply auto-fixes."""
        # Use TrustService to find violations
        from .services.trust_service import TrustService
        trust = TrustService(self.session).calculate_fleet_trust(fleet_id)
        
        remediated = []
        
        if "N2HE_MISSING" in trust["system_flags"] or "N2HE_ROTATION_DUE" in trust["system_flags"]:
            self.rotate_n2he_key(fleet_id)
            remediated.append("N2HE_ROTATED")
            
        if "CHROME_2026_INCOMPATIBLE" in trust["system_flags"]:
            # Logic to trigger EKU split is in identity_endpoints, 
            # but we can call it here too.
            pass
            
        return remediated
