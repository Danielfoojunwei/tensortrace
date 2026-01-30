"""
TensorGuard Unified Error Taxonomy.

This module provides a centralized error hierarchy for all TensorGuard components.
All errors include:
- Machine-readable error codes
- Structured details (never sensitive data)
- Request ID correlation for tracing

Error Code Naming Convention:
- TG_<COMPONENT>_<CATEGORY>_<SPECIFIC>
- Components: SDK, HE, CRYPTO, TGSP, PLATFORM
- Categories: AUTH, VALIDATION, NOT_FOUND, INTERNAL, CONFIG

Security:
- NEVER include secrets, keys, or PII in error messages
- Use placeholder IDs for tracing, not actual data
- Errors should be safe to log and return to clients
"""

from typing import Any, Dict, Optional


class TensorGuardError(Exception):
    """Base exception for all TensorGuard errors.

    All TensorGuard errors include:
    - code: Machine-readable error code (e.g., TG_HE_KEYGEN_FAILED)
    - message: Human-readable description
    - details: Structured metadata (NEVER include sensitive data)
    - request_id: Optional correlation ID for distributed tracing
    """

    def __init__(
        self,
        message: str,
        code: str = "TG_INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.request_id = request_id

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.request_id:
            parts.append(f"(request_id: {self.request_id})")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "request_id": self.request_id,
        }


# =============================================================================
# Homomorphic Encryption Errors (TG_HE_*)
# =============================================================================


class HEError(TensorGuardError):
    """Base class for homomorphic encryption errors."""

    pass


class HEKeygenError(HEError):
    """Raised when HE key generation fails."""

    def __init__(
        self,
        reason: str,
        params_hash: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"HE key generation failed: {reason}",
            code="TG_HE_KEYGEN_FAILED",
            details={"params_hash": params_hash} if params_hash else {},
            request_id=request_id,
        )


class HEEncryptionError(HEError):
    """Raised when HE encryption fails."""

    def __init__(
        self,
        reason: str,
        input_shape: Optional[tuple] = None,
        request_id: Optional[str] = None,
    ):
        details = {}
        if input_shape:
            details["input_shape"] = list(input_shape)
        super().__init__(
            message=f"HE encryption failed: {reason}",
            code="TG_HE_ENCRYPT_FAILED",
            details=details,
            request_id=request_id,
        )


class HEDecryptionError(HEError):
    """Raised when HE decryption fails."""

    def __init__(
        self,
        reason: str,
        ciphertext_type: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        details = {}
        if ciphertext_type:
            details["ciphertext_type"] = ciphertext_type
        super().__init__(
            message=f"HE decryption failed: {reason}",
            code="TG_HE_DECRYPT_FAILED",
            details=details,
            request_id=request_id,
        )


class HEParameterMismatchError(HEError):
    """Raised when HE parameters don't match between operations."""

    def __init__(
        self,
        expected_hash: str,
        actual_hash: str,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message="HE parameter mismatch between encryption and decryption context",
            code="TG_HE_PARAM_MISMATCH",
            details={
                "expected_hash": expected_hash[:16] + "...",  # Truncate for safety
                "actual_hash": actual_hash[:16] + "...",
            },
            request_id=request_id,
        )


class HELibraryNotFoundError(HEError):
    """Raised when the native HE library cannot be loaded."""

    def __init__(
        self,
        library_name: str = "N2HE",
        search_paths: Optional[list] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"{library_name} native library not found. Set {library_name.upper()}_LIB_PATH or install the library.",
            code="TG_HE_LIBRARY_NOT_FOUND",
            details={"library": library_name, "searched_paths_count": len(search_paths or [])},
            request_id=request_id,
        )


class HEToyModeError(HEError):
    """Raised when toy mode is used incorrectly."""

    def __init__(
        self,
        operation: str,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Toy HE mode attempted for operation '{operation}'. Set TENSAFE_TOY_HE=1 for non-production testing only.",
            code="TG_HE_TOY_MODE_DISABLED",
            details={"operation": operation},
            request_id=request_id,
        )


# =============================================================================
# Cryptography Errors (TG_CRYPTO_*)
# =============================================================================


class CryptoError(TensorGuardError):
    """Base class for cryptography errors."""

    pass


class CryptoDecryptionError(CryptoError):
    """Raised when authenticated decryption fails (tamper detected)."""

    def __init__(
        self,
        reason: str = "Authentication tag verification failed",
        algorithm: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Decryption failed: {reason}",
            code="TG_CRYPTO_DECRYPT_FAILED",
            details={"algorithm": algorithm} if algorithm else {},
            request_id=request_id,
        )


class CryptoKeyError(CryptoError):
    """Raised for key-related errors."""

    def __init__(
        self,
        reason: str,
        key_type: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Key error: {reason}",
            code="TG_CRYPTO_KEY_ERROR",
            details={"key_type": key_type} if key_type else {},
            request_id=request_id,
        )


class CryptoNonceReuseError(CryptoError):
    """Raised when nonce reuse is detected (critical security violation)."""

    def __init__(
        self,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message="CRITICAL: Nonce reuse detected. This is a security violation.",
            code="TG_CRYPTO_NONCE_REUSE",
            details={"severity": "critical"},
            request_id=request_id,
        )


# =============================================================================
# TGSP Package Errors (TG_TGSP_*)
# =============================================================================


class TGSPError(TensorGuardError):
    """Base class for TGSP package errors."""

    pass


class TGSPFormatError(TGSPError):
    """Raised when TGSP package format is invalid."""

    def __init__(
        self,
        reason: str,
        version: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        details = {}
        if version:
            details["version"] = version
        super().__init__(
            message=f"Invalid TGSP format: {reason}",
            code="TG_TGSP_FORMAT_ERROR",
            details=details,
            request_id=request_id,
        )


class TGSPVersionError(TGSPError):
    """Raised when TGSP version is unsupported."""

    def __init__(
        self,
        found_version: str,
        supported_versions: list,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Unsupported TGSP version: {found_version}",
            code="TG_TGSP_VERSION_UNSUPPORTED",
            details={
                "found_version": found_version,
                "supported_versions": supported_versions,
            },
            request_id=request_id,
        )


class TGSPIntegrityError(TGSPError):
    """Raised when TGSP package integrity check fails."""

    def __init__(
        self,
        component: str = "manifest",
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"TGSP integrity check failed for {component}",
            code="TG_TGSP_INTEGRITY_FAILED",
            details={"component": component},
            request_id=request_id,
        )


class TGSPRecipientError(TGSPError):
    """Raised when recipient key operations fail."""

    def __init__(
        self,
        reason: str,
        recipient_count: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        details = {}
        if recipient_count is not None:
            details["recipient_count"] = recipient_count
        super().__init__(
            message=f"TGSP recipient error: {reason}",
            code="TG_TGSP_RECIPIENT_ERROR",
            details=details,
            request_id=request_id,
        )


# =============================================================================
# Configuration Errors (TG_CONFIG_*)
# =============================================================================


class ConfigError(TensorGuardError):
    """Base class for configuration errors."""

    pass


class ConfigMissingError(ConfigError):
    """Raised when required configuration is missing."""

    def __init__(
        self,
        config_key: str,
        env_var: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        msg = f"Missing required configuration: {config_key}"
        if env_var:
            msg += f" (set {env_var})"
        super().__init__(
            message=msg,
            code="TG_CONFIG_MISSING",
            details={"config_key": config_key, "env_var": env_var} if env_var else {"config_key": config_key},
            request_id=request_id,
        )


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""

    def __init__(
        self,
        config_key: str,
        reason: str,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Invalid configuration for {config_key}: {reason}",
            code="TG_CONFIG_VALIDATION_FAILED",
            details={"config_key": config_key},
            request_id=request_id,
        )


# =============================================================================
# Error Code Registry (for documentation and validation)
# =============================================================================

ERROR_CODES = {
    # HE errors
    "TG_HE_KEYGEN_FAILED": "HE key generation failed",
    "TG_HE_ENCRYPT_FAILED": "HE encryption operation failed",
    "TG_HE_DECRYPT_FAILED": "HE decryption operation failed",
    "TG_HE_PARAM_MISMATCH": "HE parameter hash mismatch",
    "TG_HE_LIBRARY_NOT_FOUND": "Native HE library not found",
    "TG_HE_TOY_MODE_DISABLED": "Toy HE mode not enabled",
    # Crypto errors
    "TG_CRYPTO_DECRYPT_FAILED": "Authenticated decryption failed",
    "TG_CRYPTO_KEY_ERROR": "Key operation error",
    "TG_CRYPTO_NONCE_REUSE": "Nonce reuse detected (critical)",
    # TGSP errors
    "TG_TGSP_FORMAT_ERROR": "Invalid TGSP package format",
    "TG_TGSP_VERSION_UNSUPPORTED": "Unsupported TGSP version",
    "TG_TGSP_INTEGRITY_FAILED": "TGSP integrity check failed",
    "TG_TGSP_RECIPIENT_ERROR": "TGSP recipient key error",
    # Config errors
    "TG_CONFIG_MISSING": "Required configuration missing",
    "TG_CONFIG_VALIDATION_FAILED": "Configuration validation failed",
    # Internal
    "TG_INTERNAL_ERROR": "Internal error",
}


def validate_error_code(code: str) -> bool:
    """Validate that an error code is registered."""
    return code in ERROR_CODES


__all__ = [
    # Base
    "TensorGuardError",
    # HE
    "HEError",
    "HEKeygenError",
    "HEEncryptionError",
    "HEDecryptionError",
    "HEParameterMismatchError",
    "HELibraryNotFoundError",
    "HEToyModeError",
    # Crypto
    "CryptoError",
    "CryptoDecryptionError",
    "CryptoKeyError",
    "CryptoNonceReuseError",
    # TGSP
    "TGSPError",
    "TGSPFormatError",
    "TGSPVersionError",
    "TGSPIntegrityError",
    "TGSPRecipientError",
    # Config
    "ConfigError",
    "ConfigMissingError",
    "ConfigValidationError",
    # Registry
    "ERROR_CODES",
    "validate_error_code",
]
