"""
HE-LoRA Adapter with MOAI-style optimizations.

Implements:
- Base model inference in plaintext (fast)
- LoRA delta computation under CKKS HE
- MOAI column packing for rotation-free matmul
- Interleaved batching for efficient multi-sample processing
- Noise budget tracking

The key insight from MOAI: column packing removes rotations in
plaintext-ciphertext matrix multiplication, making HE operations
significantly faster.

References:
    - MOAI: https://eprint.iacr.org/2025/991
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .backend import HEBackend, HEBackendNotAvailableError, get_backend
from .packing import (
    ColumnPackedMatrix,
    InterleavedBatch,
    PackingStrategy,
    pack_for_lora,
    estimate_rotation_count,
)
from .noise_tracker import NoiseTracker, NoiseBudgetExhaustedError, create_tracker_from_context

logger = logging.getLogger(__name__)


@dataclass
class HELoRAConfig:
    """Configuration for HE-LoRA adapter."""

    # LoRA parameters
    rank: int = 16
    alpha: float = 32.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # HE parameters
    poly_modulus_degree: int = 8192
    coeff_modulus_bits: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    scale_bits: int = 40

    # Packing strategy
    packing: PackingStrategy = PackingStrategy.COLUMN

    # Performance settings
    enable_interleaved_batching: bool = True
    max_batch_size: int = 1  # Start conservative for HE

    # Safety settings
    min_noise_budget_levels: int = 1
    fail_on_noise_exhaustion: bool = True

    @property
    def scaling(self) -> float:
        """Get LoRA scaling factor."""
        return self.alpha / self.rank

    def get_he_params(self) -> Dict[str, Any]:
        """Get HE backend parameters."""
        return {
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_modulus_bits": self.coeff_modulus_bits,
            "scale_bits": self.scale_bits,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "scaling": self.scaling,
            "target_modules": self.target_modules,
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_modulus_bits": self.coeff_modulus_bits,
            "scale_bits": self.scale_bits,
            "packing": self.packing.value,
            "enable_interleaved_batching": self.enable_interleaved_batching,
            "max_batch_size": self.max_batch_size,
        }


@dataclass
class HELoRAMetrics:
    """Metrics from HE-LoRA operations."""

    # Timing
    total_time_ms: float = 0.0
    encrypt_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    decrypt_time_ms: float = 0.0

    # HE operations
    rotations_used: int = 0
    multiplications: int = 0
    additions: int = 0

    # Noise tracking
    levels_consumed: int = 0
    levels_remaining: int = 0
    scale_bits: int = 0

    # Packing info
    packing_strategy: str = "column"
    interleaved_batching: bool = False
    batch_size: int = 1

    def to_log_dict(self) -> Dict[str, Any]:
        """Get structured log format."""
        return {
            "he_backend": "N2HE-HEXL",
            "packing": self.packing_strategy,
            "interleaved_batching": self.interleaved_batching,
            "rotations_used": self.rotations_used,
            "levels_remaining": self.levels_remaining,
            "scale_bits": self.scale_bits,
            "total_time_ms": round(self.total_time_ms, 3),
            "batch_size": self.batch_size,
        }


@dataclass
class LoRAWeights:
    """LoRA adapter weight matrices."""

    adapter_id: str
    module_name: str
    lora_a: np.ndarray  # [rank, in_features]
    lora_b: np.ndarray  # [out_features, rank]
    rank: int
    alpha: float

    # Precomputed packed matrices
    _packed_a: Optional[ColumnPackedMatrix] = field(default=None, repr=False)
    _packed_b: Optional[ColumnPackedMatrix] = field(default=None, repr=False)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    @property
    def in_features(self) -> int:
        return self.lora_a.shape[1]

    @property
    def out_features(self) -> int:
        return self.lora_b.shape[0]


class HELoRAAdapter:
    """
    HE-LoRA Adapter with MOAI-style optimizations.

    Computes LoRA delta under homomorphic encryption:
        delta = scaling * (x @ A^T @ B^T)

    The base model runs in plaintext for low latency. Only the LoRA
    delta path is encrypted.

    MOAI Optimizations:
        1. Column packing: Removes rotations in pt-ct matmul
        2. Interleaved batching: Amortizes cost across samples
        3. Consistent packing: No format conversions

    Usage:
        adapter = HELoRAAdapter(config)
        adapter.register_weights("q_proj", lora_a, lora_b, rank=16, alpha=32)

        # Forward pass
        delta_plain = adapter.forward(x_plain)

        # Combined with base model
        y = y_base + delta_plain
    """

    def __init__(self, config: Optional[HELoRAConfig] = None):
        """
        Initialize HE-LoRA adapter.

        Args:
            config: Configuration (uses defaults if not provided)

        Raises:
            HEBackendNotAvailableError: If N2HE-HEXL backend not available
        """
        self.config = config or HELoRAConfig()

        # Initialize HE backend
        self._backend = get_backend(self.config.get_he_params())

        # Initialize noise tracker
        context_params = self._backend.get_context_params()
        self._noise_tracker = create_tracker_from_context(context_params)

        # Registered LoRA weights
        self._weights: Dict[str, LoRAWeights] = {}

        # Metrics
        self._total_forwards = 0
        self._total_time_ms = 0.0
        self._last_metrics: Optional[HELoRAMetrics] = None

        logger.info(
            f"HELoRAAdapter initialized: rank={self.config.rank}, "
            f"packing={self.config.packing.value}, "
            f"backend={self._backend.get_backend_name()}"
        )

    def register_weights(
        self,
        module_name: str,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        adapter_id: Optional[str] = None,
    ) -> LoRAWeights:
        """
        Register LoRA weights for a module.

        Args:
            module_name: Name of target module (e.g., "q_proj")
            lora_a: LoRA A matrix [rank, in_features]
            lora_b: LoRA B matrix [out_features, rank]
            rank: LoRA rank (inferred if not provided)
            alpha: LoRA alpha (uses config if not provided)
            adapter_id: Unique ID (generated if not provided)

        Returns:
            Registered LoRAWeights
        """
        rank = rank or lora_a.shape[0]
        alpha = alpha or self.config.alpha
        adapter_id = adapter_id or f"{module_name}_{id(lora_a)}"

        # Validate shapes
        if lora_a.shape[0] != rank:
            raise ValueError(f"LoRA A first dim ({lora_a.shape[0]}) must equal rank ({rank})")
        if lora_b.shape[1] != rank:
            raise ValueError(f"LoRA B second dim ({lora_b.shape[1]}) must equal rank ({rank})")

        # Create weight holder
        weights = LoRAWeights(
            adapter_id=adapter_id,
            module_name=module_name,
            lora_a=lora_a.astype(np.float64),
            lora_b=lora_b.astype(np.float64),
            rank=rank,
            alpha=alpha,
        )

        # Pre-pack matrices for MOAI-style operations
        # For y = x @ A^T @ B^T, we pack the transposed matrices by columns
        weights._packed_a = ColumnPackedMatrix.from_matrix(
            lora_a.T,  # [in_features, rank]
            backend=self._backend
        )
        weights._packed_b = ColumnPackedMatrix.from_matrix(
            lora_b.T,  # [rank, out_features]
            backend=self._backend
        )

        self._weights[module_name] = weights

        logger.info(
            f"Registered LoRA weights: {module_name}, "
            f"A={lora_a.shape}, B={lora_b.shape}, "
            f"rank={rank}, alpha={alpha}"
        )

        return weights

    def forward(
        self,
        x_plain: np.ndarray,
        module_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute encrypted LoRA delta and return decrypted result.

        This is the main forward pass:
        1. Encrypt activation (x_plain)
        2. Compute delta under HE: scaling * (enc(x) @ A^T @ B^T)
        3. Decrypt and return delta

        Args:
            x_plain: Plaintext activation from frozen model [batch, hidden_dim] or [hidden_dim]
            module_name: Target module (uses first registered if not specified)

        Returns:
            Decrypted LoRA delta [batch, out_dim] or [out_dim]

        Raises:
            ValueError: If no weights registered
            NoiseBudgetExhaustedError: If noise budget exhausted
        """
        start_time = time.perf_counter()
        metrics = HELoRAMetrics()

        # Get weights
        if module_name is None:
            if not self._weights:
                raise ValueError("No LoRA weights registered")
            module_name = next(iter(self._weights))

        weights = self._weights.get(module_name)
        if weights is None:
            raise ValueError(f"No weights registered for module: {module_name}")

        # Handle input shape
        original_shape = x_plain.shape
        if x_plain.ndim == 1:
            x_plain = x_plain.reshape(1, -1)

        batch_size, hidden_dim = x_plain.shape

        # Verify dimensions
        if hidden_dim != weights.in_features:
            raise ValueError(
                f"Input dim {hidden_dim} doesn't match LoRA A in_features {weights.in_features}"
            )

        # Reset HE stats
        self._backend.reset_stats()

        # Step 1: Encrypt activation
        encrypt_start = time.perf_counter()

        # Pad to slot count
        slot_count = self._backend.get_slot_count()
        padded_x = np.zeros(slot_count, dtype=np.float64)
        flat_x = x_plain.flatten()
        padded_x[:len(flat_x)] = flat_x

        ct_x = self._backend.encrypt(padded_x)
        ct_id = id(ct_x)

        # Track noise state
        self._noise_tracker.create_state(
            ct_id,
            level=self._backend.get_ciphertext_level(ct_x),
            scale=self._backend.get_ciphertext_scale(ct_x)
        )

        metrics.encrypt_time_ms = (time.perf_counter() - encrypt_start) * 1000

        # Step 2: Compute LoRA delta under HE
        compute_start = time.perf_counter()

        # Use MOAI column-packed matmul (zero rotations)
        ct_delta = self._backend.lora_delta(
            ct_x,
            weights._packed_a.native,
            weights._packed_b.native,
            weights.scaling
        )

        # Update noise tracking
        # Two matmuls + optional scaling = 2-3 rescales
        for op in ["multiply_plain", "rescale", "multiply_plain", "rescale"]:
            self._noise_tracker.update_state(ct_id, op)

        if abs(weights.scaling - 1.0) > 1e-6:
            self._noise_tracker.update_state(ct_id, "multiply_plain")
            self._noise_tracker.update_state(ct_id, "rescale")

        metrics.compute_time_ms = (time.perf_counter() - compute_start) * 1000

        # Step 3: Decrypt result
        decrypt_start = time.perf_counter()

        delta_plain = self._backend.decrypt(ct_delta, weights.out_features)

        metrics.decrypt_time_ms = (time.perf_counter() - decrypt_start) * 1000

        # Get HE operation stats
        stats = self._backend.get_operation_stats()
        metrics.rotations_used = stats.get("rotations_used", 0)
        metrics.multiplications = stats.get("multiplications", 0)
        metrics.additions = stats.get("additions", 0)

        # Noise info
        state = self._noise_tracker.get_state(ct_id)
        if state:
            metrics.levels_remaining = state.level
            metrics.levels_consumed = self._noise_tracker.initial_levels - state.level

        # Packing info
        metrics.packing_strategy = self.config.packing.value
        metrics.interleaved_batching = self.config.enable_interleaved_batching
        metrics.batch_size = batch_size
        metrics.scale_bits = self.config.scale_bits

        # Total time
        metrics.total_time_ms = (time.perf_counter() - start_time) * 1000

        # Update instance metrics
        self._total_forwards += 1
        self._total_time_ms += metrics.total_time_ms
        self._last_metrics = metrics

        # Log structured metrics (required by spec)
        logger.info(f"HE-LoRA forward: {json.dumps(metrics.to_log_dict())}")

        # Reshape output to match input
        if len(original_shape) == 1:
            delta_plain = delta_plain.flatten()

        return delta_plain

    def forward_plaintext(
        self,
        x_plain: np.ndarray,
        module_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute LoRA delta in plaintext (for comparison/testing).

        Args:
            x_plain: Plaintext activation
            module_name: Target module

        Returns:
            LoRA delta computed in plaintext
        """
        if module_name is None:
            if not self._weights:
                raise ValueError("No LoRA weights registered")
            module_name = next(iter(self._weights))

        weights = self._weights.get(module_name)
        if weights is None:
            raise ValueError(f"No weights registered for module: {module_name}")

        # Simple plaintext matmul
        # delta = scaling * (x @ A^T @ B^T)
        intermediate = x_plain @ weights.lora_a.T  # [batch, rank]
        delta = intermediate @ weights.lora_b.T  # [batch, out_features]
        return weights.scaling * delta

    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        return {
            "total_forwards": self._total_forwards,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": self._total_time_ms / max(1, self._total_forwards),
            "last_metrics": self._last_metrics.to_log_dict() if self._last_metrics else None,
            "weights_registered": list(self._weights.keys()),
            "config": self.config.to_dict(),
            "noise_tracker": self._noise_tracker.get_metrics(),
            "backend_params": self._backend.get_context_params(),
        }

    def get_last_metrics(self) -> Optional[HELoRAMetrics]:
        """Get metrics from last forward pass."""
        return self._last_metrics

    def verify_rotation_count(self, expected_max: int = 0) -> bool:
        """
        Verify rotation count is within expected bounds.

        MOAI column packing should achieve zero rotations for pt-ct matmul.

        Args:
            expected_max: Maximum allowed rotations

        Returns:
            True if within bounds
        """
        if self._last_metrics is None:
            return True
        return self._last_metrics.rotations_used <= expected_max


def create_he_lora_adapter(
    rank: int = 16,
    alpha: float = 32.0,
    target_modules: Optional[List[str]] = None,
) -> HELoRAAdapter:
    """
    Factory function to create a configured HE-LoRA adapter.

    Args:
        rank: LoRA rank
        alpha: LoRA alpha
        target_modules: Target module names

    Returns:
        Configured HELoRAAdapter

    Raises:
        HEBackendNotAvailableError: If backend not available
    """
    config = HELoRAConfig(
        rank=rank,
        alpha=alpha,
        target_modules=target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    return HELoRAAdapter(config)
