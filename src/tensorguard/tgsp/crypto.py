"""
TGSP Crypto Utilities

Provides cryptographic helper functions for TGSP package operations.
"""

import hashlib
from typing import Union


def get_sha256(data: Union[bytes, str]) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Bytes or string to hash

    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_sha256(data: Union[bytes, str], expected_hash: str) -> bool:
    """
    Verify SHA-256 hash matches expected value.

    Args:
        data: Bytes or string to verify
        expected_hash: Expected hexadecimal hash

    Returns:
        True if hash matches
    """
    return get_sha256(data) == expected_hash.lower()
