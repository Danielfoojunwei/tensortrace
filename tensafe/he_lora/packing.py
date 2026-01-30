"""
MOAI-style packing strategies for HE-LoRA.

Implements:
- Column packing for rotation-free plaintext-ciphertext matmul
- Interleaved batching for efficient multi-sample processing
- Consistent packing strategy to avoid format conversions

References:
    - MOAI: https://eprint.iacr.org/2025/991
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PackingStrategy(Enum):
    """Packing strategies for HE operations."""

    # Column packing: each column is encoded as a plaintext
    # Removes rotations in plaintext-ciphertext matmul (MOAI key optimization)
    COLUMN = "column"

    # Row packing: each row is encoded as a plaintext
    # Standard approach, requires rotations for dot products
    ROW = "row"

    # Diagonal packing: for square matrices
    # Used in ciphertext-ciphertext matmul
    DIAGONAL = "diagonal"

    # Interleaved: multiple samples packed into slots
    # Amortizes operations across batch
    INTERLEAVED = "interleaved"


@dataclass
class ColumnPackedMatrix:
    """
    Column-packed matrix for MOAI-style HE operations.

    In column packing, each column of the matrix is stored separately.
    This enables rotation-free plaintext-ciphertext multiplication:

        y = W @ x

    Where each output element is computed as:
        y[i] = sum_j(W[i,j] * x[j]) = sum_j(col_j[i] * x[j])

    Since each column is a plaintext and x is encrypted element-wise,
    we can compute this with NO rotations (MOAI's key insight).

    Attributes:
        columns: List of column vectors
        rows: Number of rows
        cols: Number of columns
        original_shape: Original matrix shape
    """

    columns: List[np.ndarray]
    rows: int
    cols: int
    original_shape: Tuple[int, int]
    _native_packed: Any = field(default=None, repr=False)

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        backend: Optional[Any] = None
    ) -> "ColumnPackedMatrix":
        """
        Create column-packed matrix from numpy array.

        Args:
            matrix: 2D numpy array to pack
            backend: Optional HE backend for native packing

        Returns:
            ColumnPackedMatrix instance
        """
        if matrix.ndim != 2:
            raise ValueError(f"Matrix must be 2D, got {matrix.ndim}D")

        rows, cols = matrix.shape
        columns = [matrix[:, j].copy() for j in range(cols)]

        packed = cls(
            columns=columns,
            rows=rows,
            cols=cols,
            original_shape=matrix.shape,
        )

        # Create native packed version if backend available
        if backend is not None:
            packed._native_packed = backend.create_column_packed_matrix(matrix)

        return packed

    def get_column(self, idx: int) -> np.ndarray:
        """Get a specific column."""
        if idx >= self.cols:
            raise IndexError(f"Column index {idx} out of range (0-{self.cols-1})")
        return self.columns[idx]

    def to_matrix(self) -> np.ndarray:
        """Convert back to dense matrix."""
        return np.column_stack(self.columns)

    @property
    def native(self) -> Any:
        """Get native packed representation."""
        if self._native_packed is None:
            raise ValueError("No native packing available. Pass backend to from_matrix().")
        return self._native_packed

    def get_packing_metadata(self) -> Dict[str, Any]:
        """Get metadata about the packing."""
        return {
            "strategy": PackingStrategy.COLUMN.value,
            "rows": self.rows,
            "cols": self.cols,
            "original_shape": self.original_shape,
            "has_native": self._native_packed is not None,
        }


@dataclass
class InterleavedBatch:
    """
    Interleaved batch packing for MOAI-style HE operations.

    Interleaved batching packs multiple samples into SIMD slots,
    amortizing the cost of HE operations across the batch.

    For a batch of vectors [x_1, x_2, ..., x_B]:
    - Each slot i contains elements from all batch samples
    - Operations are performed in parallel on all samples

    This reduces the per-sample cost and total rotations.

    Attributes:
        packed_slots: Interleaved slot data
        batch_size: Number of samples in batch
        vector_dim: Dimension of each vector
        slot_count: Total SIMD slots available
    """

    packed_slots: np.ndarray
    batch_size: int
    vector_dim: int
    slot_count: int
    _original_vectors: Optional[List[np.ndarray]] = field(default=None, repr=False)

    @classmethod
    def from_vectors(
        cls,
        vectors: List[np.ndarray],
        slot_count: int
    ) -> "InterleavedBatch":
        """
        Create interleaved batch from list of vectors.

        Args:
            vectors: List of 1D numpy arrays (same dimension)
            slot_count: Total SIMD slots available

        Returns:
            InterleavedBatch instance
        """
        if not vectors:
            raise ValueError("Must provide at least one vector")

        batch_size = len(vectors)
        vector_dim = vectors[0].shape[0]

        # Verify all vectors have same dimension
        for i, v in enumerate(vectors):
            if v.shape[0] != vector_dim:
                raise ValueError(
                    f"Vector {i} has dimension {v.shape[0]}, expected {vector_dim}"
                )

        # Check if batch fits in slots
        required_slots = batch_size * vector_dim
        if required_slots > slot_count:
            raise ValueError(
                f"Batch requires {required_slots} slots but only {slot_count} available. "
                f"Reduce batch_size ({batch_size}) or vector_dim ({vector_dim})."
            )

        # Interleave: slot[i * batch_size + b] = vectors[b][i]
        packed_slots = np.zeros(slot_count, dtype=np.float64)
        for b, vec in enumerate(vectors):
            for i in range(vector_dim):
                slot_idx = i * batch_size + b
                if slot_idx < slot_count:
                    packed_slots[slot_idx] = vec[i]

        return cls(
            packed_slots=packed_slots,
            batch_size=batch_size,
            vector_dim=vector_dim,
            slot_count=slot_count,
            _original_vectors=vectors,
        )

    def get_vector(self, batch_idx: int) -> np.ndarray:
        """
        Extract a single vector from the interleaved batch.

        Args:
            batch_idx: Index of vector in batch (0 to batch_size-1)

        Returns:
            Extracted vector
        """
        if batch_idx >= self.batch_size:
            raise IndexError(
                f"Batch index {batch_idx} out of range (0-{self.batch_size-1})"
            )

        vector = np.zeros(self.vector_dim, dtype=np.float64)
        for i in range(self.vector_dim):
            slot_idx = i * self.batch_size + batch_idx
            if slot_idx < self.slot_count:
                vector[i] = self.packed_slots[slot_idx]

        return vector

    def to_vectors(self) -> List[np.ndarray]:
        """Convert back to list of vectors."""
        return [self.get_vector(b) for b in range(self.batch_size)]

    def get_packing_metadata(self) -> Dict[str, Any]:
        """Get metadata about the packing."""
        return {
            "strategy": PackingStrategy.INTERLEAVED.value,
            "batch_size": self.batch_size,
            "vector_dim": self.vector_dim,
            "slot_count": self.slot_count,
            "utilization": (self.batch_size * self.vector_dim) / self.slot_count,
        }


def compute_optimal_batch_size(
    vector_dim: int,
    slot_count: int,
    max_batch_size: Optional[int] = None
) -> int:
    """
    Compute optimal batch size for interleaved packing.

    The optimal batch size maximizes slot utilization while
    staying within the slot count constraint.

    Args:
        vector_dim: Dimension of each vector
        slot_count: Total SIMD slots available
        max_batch_size: Optional upper limit on batch size

    Returns:
        Optimal batch size
    """
    # Maximum possible batch size
    max_possible = slot_count // vector_dim

    if max_batch_size is not None:
        max_possible = min(max_possible, max_batch_size)

    # For best efficiency, use power of 2 batch sizes
    # This helps with rotation patterns
    optimal = 1
    while optimal * 2 <= max_possible:
        optimal *= 2

    return optimal


def pack_for_lora(
    activation: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    backend: Any
) -> Tuple[np.ndarray, ColumnPackedMatrix, ColumnPackedMatrix]:
    """
    Pack activation and LoRA matrices for MOAI-style computation.

    Args:
        activation: Input activation [batch, hidden_dim] or [hidden_dim]
        lora_a: LoRA A matrix [rank, hidden_dim]
        lora_b: LoRA B matrix [out_dim, rank]
        backend: HE backend for native packing

    Returns:
        Tuple of (padded_activation, packed_A, packed_B)
    """
    # Handle batched vs unbatched input
    if activation.ndim == 1:
        activation = activation.reshape(1, -1)

    batch_size, hidden_dim = activation.shape
    rank = lora_a.shape[0]
    out_dim = lora_b.shape[0]

    # Verify dimensions
    if lora_a.shape[1] != hidden_dim:
        raise ValueError(
            f"LoRA A shape {lora_a.shape} incompatible with activation dim {hidden_dim}"
        )
    if lora_b.shape[1] != rank:
        raise ValueError(
            f"LoRA B shape {lora_b.shape} incompatible with rank {rank}"
        )

    # Pack LoRA matrices by columns
    # For y = x @ A^T @ B^T:
    # - First matmul: u = x @ A^T, need A^T column-packed
    # - Second matmul: delta = u @ B^T, need B^T column-packed
    packed_a = ColumnPackedMatrix.from_matrix(lora_a.T, backend)  # [hidden_dim, rank]
    packed_b = ColumnPackedMatrix.from_matrix(lora_b.T, backend)  # [rank, out_dim]

    # Pad activation to slot count
    slot_count = backend.get_slot_count()
    padded_activation = np.zeros(slot_count, dtype=np.float64)
    flat_activation = activation.flatten()
    padded_activation[:len(flat_activation)] = flat_activation

    logger.debug(
        f"Packed for LoRA: activation {activation.shape} -> {padded_activation.shape}, "
        f"A {lora_a.shape} -> cols {packed_a.cols}, "
        f"B {lora_b.shape} -> cols {packed_b.cols}"
    )

    return padded_activation, packed_a, packed_b


def estimate_rotation_count(
    strategy: PackingStrategy,
    matrix_shape: Tuple[int, int],
    batch_size: int = 1
) -> int:
    """
    Estimate number of rotations required for a matmul.

    Args:
        strategy: Packing strategy
        matrix_shape: Shape of weight matrix (rows, cols)
        batch_size: Batch size for interleaved packing

    Returns:
        Estimated rotation count
    """
    rows, cols = matrix_shape

    if strategy == PackingStrategy.COLUMN:
        # Column packing: ZERO rotations for pt-ct matmul
        return 0

    elif strategy == PackingStrategy.ROW:
        # Row packing: need log2(cols) rotations per row for dot product
        import math
        rotations_per_row = int(math.ceil(math.log2(cols)))
        return rows * rotations_per_row

    elif strategy == PackingStrategy.INTERLEAVED:
        # Interleaved: amortized across batch
        import math
        base_rotations = int(math.ceil(math.log2(cols)))
        return (rows * base_rotations) // batch_size

    elif strategy == PackingStrategy.DIAGONAL:
        # Diagonal: for square matrices, n rotations
        return max(rows, cols)

    return 0
