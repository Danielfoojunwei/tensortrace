# TensorGuard Error Taxonomy

**Version**: 1.0
**Date**: 2026-01-30

## Overview

TensorGuard uses a unified error taxonomy across all components. All errors include:

- **Code**: Machine-readable error code (e.g., `TG_HE_KEYGEN_FAILED`)
- **Message**: Human-readable description
- **Details**: Structured metadata (NEVER sensitive data)
- **Request ID**: Optional correlation ID for distributed tracing

## Error Code Format

```
TG_<COMPONENT>_<CATEGORY>_<SPECIFIC>
```

### Components
| Prefix | Component |
|--------|-----------|
| `TG_HE_` | Homomorphic Encryption |
| `TG_CRYPTO_` | Cryptography primitives |
| `TG_TGSP_` | TGSP package format |
| `TG_CONFIG_` | Configuration |
| `TG_SDK_` | SDK client (tg-tinker) |
| `TG_PLATFORM_` | Platform server |

## Error Code Registry

### Homomorphic Encryption Errors

| Code | Description | Actionable Steps |
|------|-------------|------------------|
| `TG_HE_KEYGEN_FAILED` | HE key generation failed | Check parameters, ensure sufficient entropy |
| `TG_HE_ENCRYPT_FAILED` | HE encryption operation failed | Verify input shape matches expected dimensions |
| `TG_HE_DECRYPT_FAILED` | HE decryption operation failed | Verify correct secret key, check ciphertext integrity |
| `TG_HE_PARAM_MISMATCH` | HE parameter hash mismatch | Ensure encryption and decryption use same parameters |
| `TG_HE_LIBRARY_NOT_FOUND` | Native HE library not found | Install N2HE or set `N2HE_LIB_PATH` |
| `TG_HE_TOY_MODE_DISABLED` | Toy HE mode not enabled | Set `TENSAFE_TOY_HE=1` for testing only |

### Cryptography Errors

| Code | Description | Actionable Steps |
|------|-------------|------------------|
| `TG_CRYPTO_DECRYPT_FAILED` | Authenticated decryption failed | Ciphertext may be tampered; verify integrity |
| `TG_CRYPTO_KEY_ERROR` | Key operation error | Check key format and validity |
| `TG_CRYPTO_NONCE_REUSE` | Nonce reuse detected (CRITICAL) | Immediately investigate; potential security breach |

### TGSP Package Errors

| Code | Description | Actionable Steps |
|------|-------------|------------------|
| `TG_TGSP_FORMAT_ERROR` | Invalid TGSP package format | Verify package was created with compatible version |
| `TG_TGSP_VERSION_UNSUPPORTED` | Unsupported TGSP version | Upgrade tools or use compatible version |
| `TG_TGSP_INTEGRITY_FAILED` | TGSP integrity check failed | Package may be corrupted or tampered |
| `TG_TGSP_RECIPIENT_ERROR` | TGSP recipient key error | Verify recipient public keys are valid |

### Configuration Errors

| Code | Description | Actionable Steps |
|------|-------------|------------------|
| `TG_CONFIG_MISSING` | Required configuration missing | Set the specified environment variable |
| `TG_CONFIG_VALIDATION_FAILED` | Configuration validation failed | Check the specified value format |

## Security Invariants

### MUST Hold

1. **Error messages NEVER contain sensitive data**
   - No keys, secrets, passwords, or tokens
   - No PII (email, names, SSN)
   - Truncate hashes to prevent reconstruction

2. **Error codes are always present**
   - Every exception has a machine-readable code
   - Codes follow the `TG_*` naming convention

3. **Request IDs enable tracing**
   - Correlation IDs flow through error hierarchy
   - Safe to log and return to clients

4. **Details are structured and safe**
   - JSON-serializable dictionary
   - No sensitive fields

### Logging

All logs are filtered for sensitive data:

```python
# These patterns trigger redaction:
SENSITIVE_PATTERNS = {
    "password", "secret", "token", "api_key",
    "private_key", "access_key", "credential"
}
```

## Usage Examples

### Raising Errors

```python
from tensorguard.errors import HEKeygenError, ConfigMissingError

# With context
raise HEKeygenError(
    reason="Insufficient entropy",
    params_hash="abc123...",
    request_id=request_id,
)

# Missing config
raise ConfigMissingError(
    config_key="N2HE_LIB_PATH",
    env_var="N2HE_LIB_PATH",
)
```

### Catching Errors

```python
from tensorguard.errors import TensorGuardError, HEError

try:
    result = he_operation()
except HEError as e:
    # Handle all HE errors
    logger.error(f"HE operation failed: {e.code}", extra=e.to_dict())
except TensorGuardError as e:
    # Handle any TensorGuard error
    logger.error(f"Operation failed: {e.code}")
```

### Structured Logging

```python
from tensorguard.logging import get_logger, configure_logging

# Configure for production
configure_logging(level="INFO", json_format=True)

# Get module logger
logger = get_logger(__name__)

# Log with context
logger.info("Processing request", extra={
    "request_id": "abc123",
    "operation": "encrypt",
    "input_shape": [32, 64],
})
```

## Testing

Run error handling tests:

```bash
pytest tests/security/test_error_handling.py -v
```

This verifies:
- All error codes are registered
- No sensitive data leakage
- Structured logging works correctly
- Error hierarchy is correct
