"""
TensorGuard Exception Hierarchy

Provides a unified exception hierarchy for all TensorGuard components.
All custom exceptions should inherit from TensorGuardError.
"""


class TensorGuardError(Exception):
    """Base exception for all TensorGuard errors."""
    pass


# === Cryptography Errors ===

class CryptographyError(TensorGuardError):
    """Raised when encryption or decryption fails."""
    pass


class KeyManagementError(CryptographyError):
    """Raised when key operations (generation, storage, retrieval) fail."""
    pass


# === Configuration Errors ===

class ConfigurationError(TensorGuardError):
    """Raised when the system is misconfigured."""
    pass


# === Communication Errors ===

class CommunicationError(TensorGuardError):
    """Raised when networking/aggregation fails."""
    pass


class TGSPClientError(CommunicationError):
    """Raised when TGSP client operations fail."""
    pass


# === Validation Errors ===

class ValidationError(TensorGuardError):
    """Raised when data validation fails."""
    pass


class ContractError(ValidationError):
    """Raised when a data contract is violated (e.g., invalid input format)."""
    pass


# === Pipeline Errors ===

class PipelineError(TensorGuardError):
    """Base class for pipeline execution errors."""
    pass


class InputError(PipelineError):
    """Raised when pipeline input acquisition fails."""
    pass


class PublishError(PipelineError):
    """Raised when artifact publishing fails."""
    pass


# === Policy Errors ===

class PolicyError(TensorGuardError):
    """Raised when policy evaluation or enforcement fails."""
    pass


class PolicyViolationError(PolicyError):
    """Raised when an action violates policy constraints."""
    pass


# === Identity Errors ===

class IdentityError(TensorGuardError):
    """Raised when identity/certificate operations fail."""
    pass


class CertificateError(IdentityError):
    """Raised when certificate operations fail."""
    pass


# === Evidence Errors ===

class EvidenceError(TensorGuardError):
    """Base class for evidence storage errors."""
    pass


class EvidenceIntegrityError(EvidenceError):
    """Raised when evidence tampering is detected."""
    pass


# === Warnings ===

class QualityWarning(UserWarning):
    """Issued when gradient quality falls below threshold."""
    pass
