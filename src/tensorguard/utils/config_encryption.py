"""
Configuration Encryption Utilities

Provides encryption/decryption for sensitive configuration fields
(e.g., API keys, credentials in integration configs).

Uses AES-256-GCM with the TG_KEY_MASTER environment variable.
"""

import os
import base64
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Check for cryptography dependency
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    logger.warning("cryptography not installed - config encryption unavailable")

from .production_gates import is_production, ProductionGateError


# Fields that should be encrypted in configs
SENSITIVE_FIELDS = frozenset({
    "api_key", "api_secret", "secret_key", "password", "token",
    "access_token", "refresh_token", "private_key", "credentials",
    "auth_token", "bearer_token", "client_secret",
})


def get_encryption_key() -> Optional[bytes]:
    """
    Get the master encryption key from environment.

    Returns:
        32-byte encryption key or None if not available
    """
    key_hex = os.environ.get("TG_KEY_MASTER")
    if not key_hex:
        if is_production():
            raise ProductionGateError(
                gate_name="CONFIG_ENCRYPTION_KEY",
                message="TG_KEY_MASTER required for config encryption in production",
                remediation="Set TG_KEY_MASTER: export TG_KEY_MASTER=$(python -c \"import os; print(os.urandom(32).hex())\")"
            )
        return None

    try:
        key = bytes.fromhex(key_hex)
        if len(key) != 32:
            raise ValueError(f"Key must be 32 bytes, got {len(key)}")
        return key
    except ValueError as e:
        if is_production():
            raise ProductionGateError(
                gate_name="CONFIG_ENCRYPTION_KEY_FORMAT",
                message=f"Invalid TG_KEY_MASTER format: {e}",
                remediation="Provide a valid 64-character hex string (32 bytes)"
            )
        logger.warning(f"Invalid TG_KEY_MASTER format: {e}")
        return None


def encrypt_value(value: str) -> str:
    """
    Encrypt a string value using AES-256-GCM.

    Args:
        value: Plaintext value to encrypt

    Returns:
        Base64-encoded encrypted value with format: "enc:v1:<nonce>:<ciphertext>"
    """
    if not HAS_CRYPTOGRAPHY:
        logger.warning("Cryptography not available - storing value unencrypted")
        return value

    key = get_encryption_key()
    if key is None:
        logger.warning("No encryption key available - storing value unencrypted")
        return value

    # Generate random nonce
    nonce = os.urandom(12)  # 96-bit nonce for GCM

    # Encrypt
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, value.encode("utf-8"), None)

    # Encode: enc:v1:<base64(nonce)>:<base64(ciphertext)>
    return f"enc:v1:{base64.b64encode(nonce).decode()}:{base64.b64encode(ciphertext).decode()}"


def decrypt_value(encrypted: str) -> str:
    """
    Decrypt an encrypted value.

    Args:
        encrypted: Encrypted value from encrypt_value()

    Returns:
        Decrypted plaintext value
    """
    # Check if actually encrypted
    if not encrypted.startswith("enc:v1:"):
        return encrypted  # Not encrypted, return as-is

    if not HAS_CRYPTOGRAPHY:
        raise ProductionGateError(
            gate_name="CONFIG_DECRYPTION",
            message="Cannot decrypt - cryptography library not installed",
            remediation="pip install cryptography>=41.0"
        )

    key = get_encryption_key()
    if key is None:
        raise ProductionGateError(
            gate_name="CONFIG_DECRYPTION_KEY",
            message="Cannot decrypt - no encryption key available",
            remediation="Set TG_KEY_MASTER environment variable"
        )

    try:
        _, version, nonce_b64, ciphertext_b64 = encrypted.split(":", 3)
        nonce = base64.b64decode(nonce_b64)
        ciphertext = base64.b64decode(ciphertext_b64)

        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to decrypt value: {e}")
        raise ValueError(f"Decryption failed: {e}")


def encrypt_sensitive_fields(config: Dict[str, Any]) -> str:
    """
    Encrypt sensitive fields in a configuration dictionary.

    Recursively scans the config and encrypts any fields whose keys
    are in SENSITIVE_FIELDS.

    Args:
        config: Configuration dictionary

    Returns:
        JSON string with sensitive fields encrypted
    """
    def _encrypt_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = _encrypt_dict(v)
            elif isinstance(v, str) and k.lower() in SENSITIVE_FIELDS:
                result[k] = encrypt_value(v)
            else:
                result[k] = v
        return result

    encrypted_config = _encrypt_dict(config)
    return json.dumps(encrypted_config)


def decrypt_sensitive_fields(config_json: str) -> Dict[str, Any]:
    """
    Decrypt sensitive fields in a configuration JSON string.

    Args:
        config_json: JSON string with potentially encrypted fields

    Returns:
        Dictionary with sensitive fields decrypted
    """
    config = json.loads(config_json)

    def _decrypt_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = _decrypt_dict(v)
            elif isinstance(v, str) and v.startswith("enc:v1:"):
                result[k] = decrypt_value(v)
            else:
                result[k] = v
        return result

    return _decrypt_dict(config)
