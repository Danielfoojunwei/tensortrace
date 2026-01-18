"""
TensorGuard Agent Identity Subsystem

This module provides the agent-side identity management for the Unified Edge Agent.
It handles the runtime lifecycle of identity operations at the edge, including:

- Certificate scanning and discovery on the local system
- CSR (Certificate Signing Request) generation
- Automated certificate deployment to local services
- TPM-based hardware attestation
- Periodic renewal checking

NOTE: This is distinct from `tensorguard.identity`, which provides the core
certificate lifecycle management library (ACME, CA, policies) used by both
the Control Plane and this agent subsystem.

Components:
- IdentityManager: Main subsystem controller for the agent daemon
- CertificateScanner: Discovers certificates from various sources
- CSRGenerator: Creates certificate signing requests with key management
- DeployerFactory: Deploys certificates to Kubernetes, Nginx, Envoy, etc.
- TPMSimulator: Hardware root of trust for attestation

Usage:
    The IdentityManager is instantiated by the AgentDaemon and runs
    as a background service when identity.enabled=True in configuration.
"""

from .manager import IdentityManager
from .scanner import CertificateScanner
from .csr_generator import CSRGenerator
from .deployers import DeployerFactory
from .tpm_simulator import TPMSimulator

__all__ = [
    "IdentityManager",
    "CertificateScanner",
    "CSRGenerator",
    "DeployerFactory",
    "TPMSimulator",
]
