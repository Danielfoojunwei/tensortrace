"""
HE-LoRA Module for TenSafe.

Provides homomorphic encryption for LoRA adapter inference with MOAI-style
optimizations:
- Column packing for rotation-free plaintext-ciphertext matmul
- Interleaved batching for efficient multi-sample processing
- Consistent packing strategy to avoid format conversions
- Noise budget tracking and level management

The base model runs in plaintext for low latency. Only the LoRA adapter
delta computation runs under HE:

    y = y_base + decrypt(HE_LoRA_Delta(encrypt(x)))

Where:
    - y_base = W_base @ x  (plaintext, fast)
    - HE_LoRA_Delta = scaling * encrypt(x) @ A^T @ B^T  (encrypted)

References:
    - MOAI: https://eprint.iacr.org/2025/991
"""

from .backend import (
    HEBackend,
    HEBackendNotAvailableError,
    get_backend,
    verify_backend,
)
from .packing import (
    ColumnPackedMatrix,
    InterleavedBatch,
    PackingStrategy,
)
from .noise_tracker import (
    NoiseTracker,
    NoiseBudgetExhaustedError,
)
from .helora_adapter import (
    HELoRAAdapter,
    HELoRAConfig,
    HELoRAMetrics,
    create_he_lora_adapter,
)

__all__ = [
    # Backend
    "HEBackend",
    "HEBackendNotAvailableError",
    "get_backend",
    "verify_backend",
    # Packing
    "ColumnPackedMatrix",
    "InterleavedBatch",
    "PackingStrategy",
    # Noise
    "NoiseTracker",
    "NoiseBudgetExhaustedError",
    # Adapter
    "HELoRAAdapter",
    "HELoRAConfig",
    "HELoRAMetrics",
    "create_he_lora_adapter",
]
