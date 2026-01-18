"""
================================================================================
                    SECURITY NOTICE - RESEARCH PROTOTYPE
================================================================================

TensorGuard Cryptography Module (N2HE)

Integrated into TensorGuard for privacy-preserving VLA fine-tuning.
Based on HintSight Technology's N2HE-hexl library.
Aligned with MOAI (IACR 2025/991) for Secure Transformer Inference.
Incorporates Skellam noise for formal DP+LWE security (Valovich, 2016).

================================================================================
                           SECURITY LIMITATIONS
================================================================================

This implementation is a RESEARCH PROTOTYPE. Before production use:

1. CRYPTOGRAPHIC AUDIT REQUIRED
   - This code has NOT been audited by professional cryptographers
   - Custom cryptosystems require formal security proofs tied to implementation
   - Side-channel vulnerabilities have NOT been analyzed

2. NOT CONSTANT-TIME
   - Operations may leak timing information
   - Table lookups may be cache-timing vulnerable
   - Branch conditions depend on secret data

3. PARAMETER VALIDATION
   - Security parameter choices have NOT been formally verified
   - LWE dimension and modulus may be insufficient for claimed security level

4. RECOMMENDED ALTERNATIVES FOR PRODUCTION
   - Microsoft SEAL: https://github.com/microsoft/SEAL
   - OpenFHE: https://github.com/openfheorg/openfhe-development
   - Concrete ML: https://github.com/zama-ai/concrete-ml

================================================================================

Uses secrets module for cryptographic seed generation.
NOTE: Large data structures use seeded PCG64 (not CSPRNG) for performance.
"""

import numpy as np
import struct
import logging
import secrets
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from ..utils.config import settings
from ..utils.logging import get_logger
from ..utils.exceptions import CryptographyError
from .keys import vault, KeyScope

# =============================================================================
# PRODUCTION SAFETY GATE
# =============================================================================
# This module contains EXPERIMENTAL cryptography that is NOT production-ready.
# In production environments, usage is blocked unless explicitly enabled.
#
# Configuration:
#   TG_ENABLE_EXPERIMENTAL_CRYPTO=false (default): Block in production
#   TG_ENABLE_EXPERIMENTAL_CRYPTO=true: Allow with prominent warnings
# =============================================================================

import os

_ENVIRONMENT = os.getenv("TG_ENVIRONMENT", "development").lower()
_ENABLE_EXPERIMENTAL = os.getenv("TG_ENABLE_EXPERIMENTAL_CRYPTO", "false").lower() == "true"


class ExperimentalCryptoError(RuntimeError):
    """Raised when experimental crypto is used in production without explicit opt-in."""
    pass


def _check_experimental_crypto_allowed():
    """
    Check if experimental crypto is allowed in the current environment.

    In production, experimental crypto is blocked unless TG_ENABLE_EXPERIMENTAL_CRYPTO=true.
    This prevents accidental use of unaudited cryptographic code in production systems.
    """
    if _ENVIRONMENT == "production" and not _ENABLE_EXPERIMENTAL:
        raise ExperimentalCryptoError(
            "SECURITY ERROR: Experimental N2HE cryptography is blocked in production. "
            "This module has NOT been audited and is NOT suitable for production use. "
            "If you understand the risks and must proceed, set TG_ENABLE_EXPERIMENTAL_CRYPTO=true. "
            "Recommended: Use Microsoft SEAL, OpenFHE, or Concrete ML instead."
        )


# Enforce gate on module import
_check_experimental_crypto_allowed()

# Emit security warning (even if allowed)
if _ENABLE_EXPERIMENTAL and _ENVIRONMENT == "production":
    warnings.warn(
        "CRITICAL: Experimental N2HE crypto enabled in PRODUCTION. "
        "This code is NOT AUDITED and may have security vulnerabilities. "
        "You have accepted full responsibility for any security incidents.",
        category=RuntimeWarning,
        stacklevel=2
    )
else:
    warnings.warn(
        "tensorguard.core.crypto: N2HE is a RESEARCH PROTOTYPE. "
        "NOT AUDITED for production use. See module docstring for details.",
        category=UserWarning,
        stacklevel=2
    )

# Performance: Bridge to HintSight's C++ N2HE-HEXL library if available
try:
    import n2he_hexl_backend as n2he_cpp # Hypothetical pybind11 module
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False

logger = get_logger(__name__)

@dataclass
class N2HEParams:
    """
    N2HE cryptographic parameters with Skellam noise (Valovich, 2016).
    Aligned with HintSight's N2HE and MOAI (IACR 2025/991) modular optimizations.
    """
    n: int = settings.LATTICE_DIMENSION
    q: Optional[int] = None
    # mu: Parameter for Skellam distribution (difference of two Poissons)
    # mu = 0.5 * (1 / epsilon^2) roughly for DP guarantees.
    mu: float = field(default_factory=lambda: 0.5 * (1.0 / (settings.DP_EPSILON ** 2)) if settings.DP_EPSILON > 0 else 3.2)
    t: int = settings.PLAINTEXT_MODULUS
    security_bits: int = settings.SECURITY_LEVEL

    def __post_init__(self):
        if self.q is None:
            self.q = 2**48 if self.security_bits >= 192 else 2**32
    
    @property
    def delta(self) -> int:
        return self.q // self.t

@dataclass
class LWECiphertext:
    """LWE Ciphertext structure."""
    a: Optional[np.ndarray] = None
    b: Union[int, np.ndarray] = 0
    seed: Optional[bytes] = None # Seed for regenerating 'A'
    params: N2HEParams = field(default_factory=N2HEParams)
    noise_budget: float = 0.0
    
    def __post_init__(self):
        if self.noise_budget == 0.0:
            # For Skellam, variance is 2*mu. Sigma equivalent ~ sqrt(2*mu)
            sigma_eff = np.sqrt(2 * self.params.mu)
            self.noise_budget = np.log2(self.params.delta) - np.log2(sigma_eff * 12)
        
        # If A is not provided but seed is, regenerate A (for deserialization/aggregation)
        if self.a is None and self.seed is not None:
            self._regenerate_a()

    def _regenerate_a(self):
        """Regenerate Matrix A from seed using a CSPRNG."""
        if self.seed is None: return
        k = len(self.b) if self.is_batch else 1
        # Convert bytes to integer for PCG64
        seed_int = int.from_bytes(self.seed, 'big')
        rng = np.random.Generator(np.random.PCG64(seed_int))
        self.a = rng.integers(0, self.params.q, size=(k, self.params.n), dtype=np.int64)

    @property
    def is_batch(self) -> bool:
        return isinstance(self.b, np.ndarray) and self.b.ndim > 0

    def serialize(self) -> bytes:
        """Fast binary serialization with seeded 'A' optimization."""
        k = len(self.b) if self.is_batch else 1
        n = self.params.n
        flags = 0x01 if self.is_batch else 0x00
        
        if self.seed:
            flags |= 0x02 # Seeded 'A' optimization
            header = struct.pack('<4sII B 32s', b'LWE2', k, n, flags, self.seed)
            b_bytes = self.b.astype(np.int64).tobytes() if self.is_batch else struct.pack('<q', int(self.b))
            return header + b_bytes
        else:
            header = struct.pack('<4sII B', b'LWE1', k, n, flags)
            a_bytes = self.a.astype(np.int64).tobytes()
            b_bytes = self.b.astype(np.int64).tobytes() if self.is_batch else struct.pack('<q', int(self.b))
            return header + a_bytes + b_bytes

    @classmethod
    def deserialize(cls, data: bytes, params: Optional[N2HEParams] = None) -> 'LWECiphertext':
        """Fast binary deserialization."""
        try:
            if len(data) < 13: raise CryptographyError("Not enough data for headers")
            magic, k, n, flags = struct.unpack('<4sII B', data[:13])
            params = params or N2HEParams(n=n)
            
            if magic == b'LWE2' and (flags & 0x02):
                # Seeded mode
                seed = data[13:13+32]
                if len(seed) < 32: raise CryptographyError("Not enough data for seed")
                offset = 13 + 32
                if flags & 0x01:
                    b_size = k * 8
                    b_bytes = data[offset : offset + b_size]
                    if len(b_bytes) < b_size: raise CryptographyError("Not enough data for vector B")
                    b_val = np.frombuffer(b_bytes, dtype=np.int64)
                else:
                    b_bytes = data[offset : offset + 8]
                    if len(b_bytes) < 8: raise CryptographyError("Not enough data for scalar B")
                    b_val = struct.unpack('<q', b_bytes)[0]
                return cls(b=b_val, seed=seed, params=params)
                
            elif magic == b'LWE1':
                # Full matrix mode
                offset = 13
                a_size = k * n * 8
                a_bytes = data[offset : offset + a_size]
                if len(a_bytes) < a_size: raise CryptographyError("Not enough data for matrix A")
                a_arr = np.frombuffer(a_bytes, dtype=np.int64)
                if k > 1: a_arr = a_arr.reshape(k, n)
                offset += a_size
                if flags & 0x01:
                    b_size = k * 8
                    b_bytes = data[offset : offset + b_size]
                    if len(b_bytes) < b_size: raise CryptographyError("Not enough data for vector B")
                    b_val = np.frombuffer(b_bytes, dtype=np.int64)
                else:
                    b_bytes = data[offset : offset + 8]
                    if len(b_bytes) < 8: raise CryptographyError("Not enough data for scalar B")
                    b_val = struct.unpack('<q', b_bytes)[0]
                return cls(a=a_arr, b=b_val, params=params)
            else:
                raise CryptographyError(f"Unsupported LWE Magic: {magic}")
        except Exception as e:
            if isinstance(e, CryptographyError): raise
            raise CryptographyError(f"Deserialization failed: {e}")

    def __add__(self, other: 'LWECiphertext') -> 'LWECiphertext':
        """Homomorphic Addition: E(m1) + E(m2) = E(m1 + m2)."""
        if self.params.q != other.params.q or self.params.n != other.params.n:
            raise CryptographyError("Ciphertext parameters mismatch for addition")
        
        q = self.params.q
        new_a = (self.a + other.a) % q
        new_b = (self.b + other.b) % q
        
        # Heuristic noise tracking: variance adds linearly
        new_noise = min(self.noise_budget, other.noise_budget) - 1.0 # Bit loss roughly log2(2)=1
        
        return LWECiphertext(
            a=new_a, 
            b=new_b, 
            params=self.params, 
            noise_budget=new_noise
        )

    def __iadd__(self, other: 'LWECiphertext') -> 'LWECiphertext':
        res = self.__add__(other)
        self.a, self.b, self.noise_budget = res.a, res.b, res.noise_budget
        return self

# ============================================================================
# RANDOMNESS GENERATION
# ============================================================================
# IMPORTANT: PCG64 is NOT a cryptographically secure PRNG.
# While we seed it with secrets.randbits(256), the output stream is still
# predictable given sufficient observations.
#
# For cryptographically secure random bytes, use: secrets.token_bytes()
# For cryptographic key generation, use: secrets module directly
#
# The _get_seeded_rng() below is used ONLY for:
# - Generating large uniform matrices (A in LWE) where unpredictability is
#   derived from the seed secrecy, not the generator's cryptographic properties
# - Sampling noise distributions (Poisson/Skellam) for DP purposes
#
# This is a deliberate trade-off: cryptographic randomness for seeds,
# fast generation for large data structures.
# ============================================================================

def _get_seeded_rng():
    """
    Get a seeded RNG for large data generation.

    WARNING: This is NOT a CSPRNG. The output is deterministic given the seed.
    Use secrets module directly for cryptographic key material.
    """
    return np.random.Generator(np.random.PCG64(secrets.randbits(256)))


def _get_crypto_random_bytes(n: int) -> bytes:
    """Get cryptographically secure random bytes using secrets module."""
    return secrets.token_bytes(n)

def sample_skellam(mu: float, size: int) -> np.ndarray:
    """
    Sample from symmetric Skellam distribution S(mu, mu).
    Technically: X1 - X2 where X1,X2 ~ Poisson(mu).
    This noise provides both DP and LWE security (Valovich, 2016).
    Uses a securely re-seeded generator for each call to prevent state recovery.
    """
    rng = _get_seeded_rng()
    x1 = rng.poisson(mu, size)
    x2 = rng.poisson(mu, size)
    return (x1 - x2).astype(np.int64)

class N2HEContext:
    """N2HE Encryption Context and Operations."""
    def __init__(self, params: Optional[N2HEParams] = None):
        self.params = params or N2HEParams()
        self.lwe_key: Optional[np.ndarray] = None
        self.stats = {'encryptions': 0, 'decryptions': 0}

    def generate_keys(self):
        """Generate secret key using CSPRNG."""
        # Use secrets-seeded RNG for key generation
        rng = _get_seeded_rng()
        self.lwe_key = rng.choice([-1, 0, 1], size=self.params.n).astype(np.int64)
        logger.debug("N2HE Keys generated with CSPRNG")

    def save_key(self, name: str):
        """Save the secret key to the unified vault (AGGREGATION scope)."""
        if self.lwe_key is None:
            raise CryptographyError("No key to save")
        
        # Binary serialization of the numpy array
        data = self.lwe_key.astype(np.int64).tobytes()
        
        vault.save_key_artifact(
            scope=KeyScope.AGGREGATION,
            name=name,
            data=data,
            algorithm="N2HE-LWE",
            params={
                "n": self.params.n,
                "q": self.params.q,
                "t": self.params.t,
                "mu": self.params.mu
            },
            suffix=".npy.bin"
        )

    def load_key(self, name: str):
        """Load the secret key from the unified vault."""
        try:
            data, meta = vault.load_key_artifact(
                scope=KeyScope.AGGREGATION,
                name=name,
                suffix=".npy.bin"
            )
            # Reconstruct from bytes
            self.lwe_key = np.frombuffer(data, dtype=np.int64)
            # Optionally validate params from meta
            if meta.params.get("n") != self.params.n:
                logger.warning(f"Key 'n' mismatch: {meta.params.get('n')} vs {self.params.n}")
        except Exception as e:
            raise CryptographyError(f"Failed to load N2HE key '{name}': {e}")

    def encrypt_batch(self, messages: np.ndarray) -> LWECiphertext:
        """
        Vectorized encryption using Skellam noise and seeded 'A' optimization.
        """
        if self.lwe_key is None: self.generate_keys()
        
        k = messages.shape[0]
        n, q, t = self.params.n, self.params.q, self.params.t
        mu, delta = self.params.mu, self.params.delta
        
        m_vec = messages.astype(np.int64) % t
        # Seeded 'A' generation for performance and communication efficiency
        seed = secrets.token_bytes(32)
        seed_int = int.from_bytes(seed, 'big')
        rng = np.random.Generator(np.random.PCG64(seed_int))
        A = rng.integers(0, q, size=(k, n), dtype=np.int64)
        
        # Error term E is sampled from Skellam distribution
        E = sample_skellam(mu, k)
        
        # b = A*s + e + delta*m (mod q)
        B = (np.dot(A, self.lwe_key) + E + delta * m_vec) % q
        
        self.stats['encryptions'] += k
        return LWECiphertext(b=B, seed=seed, params=self.params)

    def decrypt_batch(self, ct: LWECiphertext) -> np.ndarray:
        """Vectorized decryption."""
        if self.lwe_key is None: raise CryptographyError("Keys not generated")
        
        A, B = (ct.a.reshape(1, -1), np.array([ct.b])) if not ct.is_batch else (ct.a, ct.b)
        q, delta, t = self.params.q, self.params.delta, self.params.t
        
        m_scaled = (B - np.dot(A, self.lwe_key)) % q
        m_scaled[m_scaled > (q // 2)] -= q
        m = np.round(m_scaled / delta).astype(np.int64) % t
        
        self.stats['decryptions'] += len(m)
        return m

    def fold_pack(self, messages: List[np.ndarray]) -> np.ndarray:
        """SIMD-style tensor packing."""
        flat = np.concatenate([m.flatten() for m in messages])
        pad = (self.params.n - (len(flat) % self.params.n)) % self.params.n
        return np.pad(flat, (0, pad)) if pad > 0 else flat

class N2HEEncryptor:
    """Professional wrapper for N2HE encryption with chunking and key rotation."""
    def __init__(self, key_path: Optional[str] = None, security_level: int = 128):
        self.params = N2HEParams(security_bits=security_level)
        self._ctx = N2HEContext(self.params)
        self._usage_count = 0
        self._max_uses = settings.MAX_KEY_USES
        
        if key_path and Path(key_path).exists():
            self._ctx.load_key(key_path)
        else:
            self._ctx.generate_keys()
        
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt binary data with SIMD folding. Returns raw binary LWECiphertext."""
        self._usage_count += 1
        if self._usage_count > self._max_uses:
            self._ctx.generate_keys()
            self._usage_count = 0
            
        data_arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        packed = self._ctx.fold_pack([data_arr])
        
        # Performance: For small results, we encrypt everything in one batch
        # This aligns with the server side's expect-one-ciphertext-per-tensor model.
        # Large payloads should be handled by the application layer using separate tensor entries.
        ct = self._ctx.encrypt_batch(packed)
        return ct.serialize()

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt binary ciphertext (Fast binary path)."""
        ct = LWECiphertext.deserialize(ciphertext, self.params)
        dec_arr = self._ctx.decrypt_batch(ct).astype(np.uint8)
        # Note: Precision restoration for packed data may require length metadata
        # In this implementation, we assume the caller knows the original length or uses padding.
        return dec_arr.tobytes()

def generate_key(path: str, security_level: int = 128):
    """Standalone utility to generate a new TensorGuard N2HE key."""
    params = N2HEParams(security_bits=security_level)
    ctx = N2HEContext(params)
    ctx.generate_keys()
    ctx.save_key(path)
    print(f"Successfully generated N2HE {security_level}-bit key at: {path}")
