"""
HE Backend wrapper for TenSafe HE-LoRA.

This module provides a thin abstraction over the N2HE-HEXL backend,
enforcing that real HE is used (no toy/simulation fallback).
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HEBackendNotAvailableError(Exception):
    """Raised when the HE backend is not available or not properly installed."""

    def __init__(self, message: Optional[str] = None):
        default_msg = (
            "N2HE-HEXL backend is required but not available.\n"
            "This is a production requirement - no toy/simulation fallback exists.\n"
            "\n"
            "To install:\n"
            "    ./scripts/build_n2he_hexl.sh\n"
            "\n"
            "Then verify:\n"
            "    python scripts/verify_he_backend.py"
        )
        super().__init__(message or default_msg)


class HEBackend:
    """
    HE Backend abstraction for TenSafe HE-LoRA.

    Wraps the N2HE-HEXL backend with additional validation and logging.
    Enforces that real HE is used - no toy mode allowed.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize HE backend.

        Args:
            params: Optional CKKS parameters dict with keys:
                - poly_modulus_degree: Ring degree (default: 8192)
                - coeff_modulus_bits: Modulus chain (default: [60, 40, 40, 60])
                - scale_bits: Fixed-point scale (default: 40)
        """
        # Import here to fail fast if not available
        try:
            from crypto_backend.n2he_hexl import (
                N2HEHEXLBackend,
                CKKSParams,
                CKKSCiphertext,
                ColumnPackedMatrix,
            )
        except ImportError as e:
            raise HEBackendNotAvailableError(
                f"Failed to import N2HE-HEXL backend: {e}"
            ) from e

        # Configure parameters
        ckks_params = CKKSParams()
        if params:
            if "poly_modulus_degree" in params:
                ckks_params.poly_modulus_degree = params["poly_modulus_degree"]
            if "coeff_modulus_bits" in params:
                ckks_params.coeff_modulus_bits = params["coeff_modulus_bits"]
            if "scale_bits" in params:
                ckks_params.scale_bits = params["scale_bits"]

        # Initialize backend
        self._backend = N2HEHEXLBackend(ckks_params)
        self._CKKSCiphertext = CKKSCiphertext
        self._ColumnPackedMatrix = ColumnPackedMatrix
        self._setup_complete = False
        self._keys_generated = False

        logger.info(f"HE Backend initialized with params: {params or 'default'}")

    def setup(self) -> None:
        """Set up CKKS context and generate keys."""
        if self._setup_complete:
            return

        self._backend.setup_context()
        self._backend.generate_keys(generate_galois=True)

        self._setup_complete = True
        self._keys_generated = True

        params = self._backend.get_context_params()
        logger.info(
            f"HE Backend setup complete: "
            f"ring_degree={params['ring_degree']}, "
            f"slot_count={params['slot_count']}, "
            f"galois_keys={params['has_galois_keys']}"
        )

    def is_available(self) -> bool:
        """Check if backend is available and functional."""
        return self._backend.is_available()

    def is_ready(self) -> bool:
        """Check if backend is ready for operations."""
        return self._setup_complete and self._keys_generated

    def get_backend_name(self) -> str:
        """Get the backend implementation name."""
        return self._backend.get_backend_name()

    def get_context_params(self) -> Dict[str, Any]:
        """Get CKKS context parameters."""
        if not self._setup_complete:
            return {}
        return self._backend.get_context_params()

    def get_slot_count(self) -> int:
        """Get number of SIMD slots."""
        params = self.get_context_params()
        return params.get("slot_count", 0)

    def encrypt(self, plaintext: np.ndarray) -> Any:
        """
        Encrypt a plaintext vector.

        Args:
            plaintext: 1D numpy array of floats

        Returns:
            CKKSCiphertext object
        """
        if not self.is_ready():
            raise RuntimeError("Backend not ready. Call setup() first.")

        if plaintext.ndim != 1:
            plaintext = plaintext.flatten()

        return self._backend.encrypt(plaintext.astype(np.float64))

    def decrypt(self, ciphertext: Any, output_size: int = 0) -> np.ndarray:
        """
        Decrypt a ciphertext to plaintext vector.

        Args:
            ciphertext: CKKSCiphertext object
            output_size: Number of elements to return (0 = all)

        Returns:
            1D numpy array of floats
        """
        if not self.is_ready():
            raise RuntimeError("Backend not ready. Call setup() first.")

        return self._backend.decrypt(ciphertext, output_size)

    def create_column_packed_matrix(self, matrix: np.ndarray) -> Any:
        """
        Create a column-packed matrix for MOAI-style operations.

        Args:
            matrix: 2D numpy array (will be packed by columns)

        Returns:
            ColumnPackedMatrix object
        """
        return self._ColumnPackedMatrix(matrix.astype(np.float64))

    def column_packed_matmul(
        self,
        ct_x: Any,
        weight: Any,
        rescale: bool = True
    ) -> Any:
        """
        MOAI-style column-packed plaintext-ciphertext matmul.

        This achieves ZERO rotations for the operation.

        Args:
            ct_x: Encrypted input vector
            weight: ColumnPackedMatrix
            rescale: Whether to rescale after operation

        Returns:
            Encrypted result
        """
        return self._backend.column_packed_matmul(ct_x, weight, rescale)

    def lora_delta(
        self,
        ct_x: Any,
        lora_a: Any,
        lora_b: Any,
        scaling: float = 1.0
    ) -> Any:
        """
        Compute LoRA delta with MOAI column packing.

        delta = scaling * (x @ A^T @ B^T)

        Args:
            ct_x: Encrypted activation
            lora_a: ColumnPackedMatrix for A
            lora_b: ColumnPackedMatrix for B
            scaling: LoRA scaling factor

        Returns:
            Encrypted delta
        """
        return self._backend.lora_delta(ct_x, lora_a, lora_b, scaling)

    def get_operation_stats(self) -> Dict[str, int]:
        """Get operation statistics (rotations, multiplications, etc.)."""
        return self._backend.get_operation_stats()

    def reset_stats(self) -> None:
        """Reset operation counters."""
        self._backend.reset_stats()

    def get_ciphertext_level(self, ciphertext: Any) -> int:
        """Get remaining multiplicative levels."""
        return ciphertext.level

    def get_ciphertext_scale(self, ciphertext: Any) -> float:
        """Get ciphertext scale."""
        return ciphertext.scale


def get_backend(params: Optional[Dict[str, Any]] = None) -> HEBackend:
    """
    Get a configured HE backend instance.

    This is the primary factory function for getting HE backends.
    It enforces that the real N2HE-HEXL backend is used.

    Args:
        params: Optional CKKS parameters

    Returns:
        Configured HEBackend instance

    Raises:
        HEBackendNotAvailableError: If N2HE-HEXL is not installed
    """
    backend = HEBackend(params)
    backend.setup()
    return backend


def verify_backend() -> Dict[str, Any]:
    """
    Verify the HE backend is properly installed and functional.

    Returns:
        Verification results dict

    Raises:
        HEBackendNotAvailableError: If verification fails
    """
    try:
        from crypto_backend.n2he_hexl import verify_backend as _verify
        return _verify()
    except ImportError as e:
        raise HEBackendNotAvailableError(
            f"Cannot verify backend: {e}"
        ) from e
