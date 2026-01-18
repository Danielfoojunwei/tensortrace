"""
TensorGuard Identity Module (Core Library)

Machine Identity Guard - Certificate Lifecycle Management for Zero-Trust Robotics.

This is the CORE LIBRARY for certificate lifecycle management, providing:
- Certificate inventory and discovery services
- Policy-driven automated renewal scheduling
- ACME integration (Let's Encrypt, ZeroSSL)
- Private CA for mTLS/client auth
- Tamper-evident audit logging

NOTE: This module is used by both the Control Plane (for centralized management)
and the Edge Agent (via tensorguard.agent.identity). The agent-side module
`tensorguard.agent.identity` provides the runtime identity subsystem that
uses this core library for actual operations.

Architecture:
- Control Plane: Uses this module directly for fleet-wide identity management
- Edge Agent: Uses tensorguard.agent.identity.IdentityManager, which
  coordinates with the Control Plane and performs local operations
"""

__version__ = "2.1.0"

# Core services
from .policy_engine import PolicyEngine
from .inventory import InventoryService
from .audit import AuditService
from .scheduler import RenewalScheduler

# ACME
from .acme.client import ACMEClient

# Private CA
from .ca.private_ca import PrivateCAClient

# Keys
from .keys.provider import KeyProvider, FileKeyProvider

__all__ = [
    "PolicyEngine",
    "InventoryService", 
    "AuditService",
    "RenewalScheduler",
    "ACMEClient",
    "PrivateCAClient",
    "KeyProvider",
    "FileKeyProvider",
]
