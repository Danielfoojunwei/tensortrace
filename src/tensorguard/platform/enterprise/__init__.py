"""
Proprietary Enterprise Extensions (Stubs)
This module marks the boundary for Enterprise features.

SECURITY NOTE: Community edition behavior is configurable.
- TG_COMMUNITY_MODE=true (default): Enterprise features are DISABLED (fail-closed)
- TG_COMMUNITY_MODE=permissive: Enterprise features allowed for local dev/testing only
"""
import os
import logging

logger = logging.getLogger(__name__)

# Enterprise feature list - features that require entitlement
ENTERPRISE_FEATURES = {
    "advanced_audit",
    "sso_integration",
    "multi_tenant",
    "custom_policy_engine",
    "hsm_integration",
    "compliance_export",
    "priority_support",
}

# Community features - always available
COMMUNITY_FEATURES = {
    "basic_auth",
    "single_tenant",
    "basic_audit",
    "tgsp_packaging",
    "evidence_fabric",
}

_COMMUNITY_MODE = os.getenv("TG_COMMUNITY_MODE", "true").lower()
_ENVIRONMENT = os.getenv("TG_ENVIRONMENT", "development")


def check_entitlement(user: str, feature: str) -> bool:
    """
    Check if user/tenant has entitlement for a feature.

    Community Edition behavior (fail-closed for enterprise features):
    - Community features: Always allowed
    - Enterprise features: Denied unless TG_COMMUNITY_MODE=permissive in dev
    """
    # Community features are always allowed
    if feature in COMMUNITY_FEATURES:
        return True

    # Enterprise features require explicit entitlement
    if feature in ENTERPRISE_FEATURES:
        if _COMMUNITY_MODE == "permissive" and _ENVIRONMENT != "production":
            logger.warning(
                f"COMMUNITY MODE (permissive): Allowing enterprise feature '{feature}' "
                f"for development. This would be denied in production."
            )
            return True
        else:
            logger.info(f"Enterprise feature '{feature}' denied - not entitled")
            return False

    # Unknown features default to denied (fail-closed)
    logger.warning(f"Unknown feature requested: '{feature}' - denying (fail-closed)")
    return False


def log_audit_event(event: dict):
    """
    Log an audit event.

    Community Edition: Logs to standard logger (no persistent audit trail).
    Enterprise Edition: Would persist to audit store with tamper-evidence.
    """
    logger.info(f"AUDIT: {event}")
    # Community mode: No persistent audit store
    # Enterprise mode would: store to DB, sign event, replicate to audit cluster
