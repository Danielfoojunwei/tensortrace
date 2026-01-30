"""
TG-Tinker: Privacy-First ML Training Platform

A complete privacy-preserving machine learning training system featuring:
- Differential Privacy (DP-SGD) with RDP/Moments/PRV accountants
- AES-256-GCM encrypted artifact storage with per-tenant isolation
- Hash-chained tamper-evident audit logging
- TGSP secure packaging with hybrid PQC signatures (Ed25519 + Dilithium3)
- Async training API with FutureHandle pattern
"""

__version__ = "3.0.0"
__author__ = "Daniel Foo & The TG-Tinker Team"

# Core cryptographic components
from .crypto import (
    sign_hybrid,
    verify_hybrid,
    generate_hybrid_keypair,
)

# TGSP secure packaging
from .tgsp import TGSPService

# Edge client for TGSP
from .edge import TGSPEdgeClient

__all__ = [
    # Version
    "__version__",
    # Crypto
    "sign_hybrid",
    "verify_hybrid",
    "generate_hybrid_keypair",
    # TGSP
    "TGSPService",
    # Edge
    "TGSPEdgeClient",
]
