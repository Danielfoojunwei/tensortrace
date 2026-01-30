"""
ML-KEM-768 (Kyber-768) Post-Quantum Key Encapsulation Mechanism

This module provides ML-KEM-768 key encapsulation using liboqs (Open Quantum Safe).
ML-KEM-768 is standardized in NIST FIPS 203.

Requirements:
    - liboqs native library: https://github.com/open-quantum-safe/liboqs
    - liboqs-python: pip install liboqs-python

liboqs is required for all cryptographic operations.
"""

import logging
from typing import Tuple

from .agility import PostQuantumKEM

logger = logging.getLogger(__name__)

# Try to import liboqs
_LIBOQS_AVAILABLE = False
_oqs = None

try:
    import oqs

    _oqs = oqs
    _LIBOQS_AVAILABLE = True
    logger.info("liboqs loaded successfully - using production PQC")
except ImportError:
    logger.error("liboqs not available. Install with: pip install liboqs-python (requires liboqs native library).")


class Kyber768(PostQuantumKEM):
    """
    ML-KEM-768 (Kyber-768) Key Encapsulation Mechanism.

    This implementation uses liboqs for production-grade post-quantum security.

    Security Level: NIST Level 3 (equivalent to AES-192)
    Standard: NIST FIPS 203 (ML-KEM)

    Example:
        kem = Kyber768()
        pk, sk = kem.keygen()
        shared_secret, ciphertext = kem.encap(pk)
        recovered_secret = kem.decap(sk, ciphertext)
        assert shared_secret == recovered_secret
    """

    NAME = "ML-KEM-768"

    # NIST ML-KEM-768 sizes
    PK_SIZE = 1184
    SK_SIZE = 2400
    CT_SIZE = 1088
    SS_SIZE = 32

    def __init__(self):
        """Initialize Kyber-768 KEM."""
        if not _LIBOQS_AVAILABLE:
            raise ImportError("Kyber768 requires liboqs. Install liboqs-python with the liboqs native library.")
        self._kem = _oqs.KeyEncapsulation("ML-KEM-768")
        logger.debug("Kyber768 initialized with liboqs ML-KEM-768")

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def public_key_size(self) -> int:
        return self.PK_SIZE

    @property
    def ciphertext_size(self) -> int:
        return self.CT_SIZE

    @property
    def is_production(self) -> bool:
        """Returns True if using real liboqs implementation."""
        return True

    def keygen(self) -> Tuple[bytes, bytes]:
        """
        Generate a new ML-KEM-768 keypair.

        Returns:
            Tuple of (public_key, secret_key)
        """
        kem = _oqs.KeyEncapsulation("ML-KEM-768")
        pk = kem.generate_keypair()
        sk = kem.export_secret_key()
        return bytes(pk), bytes(sk)

    def encap(self, pk: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate a shared secret using the recipient's public key.

        Args:
            pk: Recipient's public key

        Returns:
            Tuple of (shared_secret, ciphertext)
        """
        if len(pk) != self.PK_SIZE:
            raise ValueError(f"Invalid public key size: expected {self.PK_SIZE}, got {len(pk)}")

        kem = _oqs.KeyEncapsulation("ML-KEM-768")
        ciphertext, shared_secret = kem.encap_secret(pk)
        return bytes(shared_secret), bytes(ciphertext)

    def decap(self, sk: bytes, ct: bytes) -> bytes:
        """
        Decapsulate a shared secret using the secret key.

        Args:
            sk: Secret key
            ct: Ciphertext from encapsulation

        Returns:
            Shared secret
        """
        if len(sk) != self.SK_SIZE:
            raise ValueError(f"Invalid secret key size: expected {self.SK_SIZE}, got {len(sk)}")
        if len(ct) != self.CT_SIZE:
            raise ValueError(f"Invalid ciphertext size: expected {self.CT_SIZE}, got {len(ct)}")

        kem = _oqs.KeyEncapsulation("ML-KEM-768", sk)
        shared_secret = kem.decap_secret(ct)
        return bytes(shared_secret)


def is_liboqs_available() -> bool:
    """Check if liboqs is available for production PQC."""
    return _LIBOQS_AVAILABLE
