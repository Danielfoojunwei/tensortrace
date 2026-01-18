"""
Safe Serialization Utilities

This module provides safe serialization alternatives to pickle.
NEVER use pickle for untrusted data - it allows arbitrary code execution.

Supported formats:
- msgpack: Fast binary serialization (recommended for internal use)
- JSON: Human-readable, interoperable (recommended for APIs)

For numpy arrays, use numpy's native .npy format or convert to lists.
"""

import json
import warnings
from typing import Any, Union
from dataclasses import asdict, is_dataclass

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    warnings.warn(
        "msgpack not installed. Install with: pip install msgpack. "
        "Falling back to JSON for binary serialization.",
        UserWarning
    )

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class SerializationError(Exception):
    """Raised when serialization or deserialization fails."""
    pass


def _encode_numpy(obj: Any) -> Any:
    """Convert numpy arrays to serializable format."""
    if HAS_NUMPY and isinstance(obj, np.ndarray):
        return {
            "__numpy__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tobytes().hex()
        }
    return obj


def _decode_numpy(obj: Any) -> Any:
    """Restore numpy arrays from serialized format."""
    if isinstance(obj, dict) and obj.get("__numpy__"):
        if not HAS_NUMPY:
            raise SerializationError("numpy required to deserialize numpy arrays")
        dtype = np.dtype(obj["dtype"])
        shape = tuple(obj["shape"])
        data = bytes.fromhex(obj["data"])
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    return obj


def _prepare_for_serialization(obj: Any) -> Any:
    """Recursively prepare object for serialization."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {"__dataclass__": type(obj).__name__, "data": asdict(obj)}
    elif isinstance(obj, dict):
        return {k: _prepare_for_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_prepare_for_serialization(v) for v in obj]
    elif isinstance(obj, bytes):
        return {"__bytes__": True, "data": obj.hex()}
    elif HAS_NUMPY and isinstance(obj, np.ndarray):
        return _encode_numpy(obj)
    return obj


def _restore_from_serialization(obj: Any) -> Any:
    """Recursively restore object from serialized format."""
    if isinstance(obj, dict):
        if obj.get("__bytes__"):
            return bytes.fromhex(obj["data"])
        elif obj.get("__numpy__"):
            return _decode_numpy(obj)
        else:
            return {k: _restore_from_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_restore_from_serialization(v) for v in obj]
    return obj


def safe_dumps(obj: Any) -> bytes:
    """
    Safely serialize object to bytes using msgpack.

    This is the recommended method for internal binary serialization.
    Unlike pickle, msgpack cannot execute arbitrary code.

    Args:
        obj: Object to serialize (must be JSON-compatible types + numpy arrays)

    Returns:
        Serialized bytes
    """
    prepared = _prepare_for_serialization(obj)

    if HAS_MSGPACK:
        return msgpack.packb(prepared, use_bin_type=True)
    else:
        # Fallback to JSON
        return json.dumps(prepared, sort_keys=True).encode('utf-8')


def safe_loads(data: bytes) -> Any:
    """
    Safely deserialize bytes to object using msgpack.

    This is safe for untrusted data - it cannot execute arbitrary code.

    Args:
        data: Bytes to deserialize

    Returns:
        Deserialized object
    """
    if HAS_MSGPACK:
        raw = msgpack.unpackb(data, raw=False)
    else:
        raw = json.loads(data.decode('utf-8'))

    return _restore_from_serialization(raw)


def safe_dump(obj: Any, path: str) -> None:
    """
    Safely serialize object to file.

    Args:
        obj: Object to serialize
        path: File path to write to
    """
    data = safe_dumps(obj)
    with open(path, 'wb') as f:
        f.write(data)


def safe_load(path: str) -> Any:
    """
    Safely deserialize object from file.

    Args:
        path: File path to read from

    Returns:
        Deserialized object
    """
    with open(path, 'rb') as f:
        data = f.read()
    return safe_loads(data)


# Compatibility aliases for code migrating from pickle
dumps = safe_dumps
loads = safe_loads
dump = safe_dump
load = safe_load
