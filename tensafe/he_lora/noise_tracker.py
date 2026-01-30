"""
Noise budget and level tracking for HE-LoRA operations.

Tracks:
- Remaining multiplicative levels
- Estimated noise budget
- Scale management
- Operation costs

This ensures we don't exhaust the noise budget during LoRA computation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NoiseBudgetExhaustedError(Exception):
    """Raised when noise budget is exhausted and computation cannot continue."""

    def __init__(
        self,
        message: str,
        remaining_levels: int = 0,
        required_levels: int = 0
    ):
        super().__init__(message)
        self.remaining_levels = remaining_levels
        self.required_levels = required_levels


@dataclass
class OperationCost:
    """Cost of an HE operation in terms of levels and noise."""

    name: str
    levels_consumed: int
    estimated_noise_bits: float
    operation_type: str  # "multiply", "add", "rotate", "rescale"


# Standard operation costs for CKKS
OPERATION_COSTS = {
    "multiply_plain": OperationCost("multiply_plain", levels_consumed=0, estimated_noise_bits=1.0, operation_type="multiply"),
    "multiply_cipher": OperationCost("multiply_cipher", levels_consumed=1, estimated_noise_bits=2.0, operation_type="multiply"),
    "add": OperationCost("add", levels_consumed=0, estimated_noise_bits=0.5, operation_type="add"),
    "rotate": OperationCost("rotate", levels_consumed=0, estimated_noise_bits=1.0, operation_type="rotate"),
    "rescale": OperationCost("rescale", levels_consumed=1, estimated_noise_bits=0.0, operation_type="rescale"),
    "relinearize": OperationCost("relinearize", levels_consumed=0, estimated_noise_bits=1.5, operation_type="multiply"),
}


@dataclass
class NoiseState:
    """Current noise state of a ciphertext."""

    level: int  # Current level in modulus chain
    scale: float  # Current scale
    estimated_noise_bits: float  # Estimated noise (bits)
    operations: List[str] = field(default_factory=list)  # Operation history


class NoiseTracker:
    """
    Tracks noise budget and levels for HE operations.

    Monitors the noise state of ciphertexts through operations and
    raises an error if the budget would be exhausted.

    Attributes:
        initial_levels: Total levels in modulus chain
        min_levels_required: Minimum levels to reserve (for decryption)
        noise_threshold_bits: Noise threshold before warning
    """

    def __init__(
        self,
        initial_levels: int,
        scale_bits: int = 40,
        min_levels_required: int = 1,
        noise_threshold_bits: float = 5.0,
    ):
        """
        Initialize noise tracker.

        Args:
            initial_levels: Total levels in coefficient modulus chain
            scale_bits: Bits used for scale
            min_levels_required: Minimum levels to preserve
            noise_threshold_bits: Warn when noise exceeds this
        """
        self.initial_levels = initial_levels
        self.scale_bits = scale_bits
        self.min_levels_required = min_levels_required
        self.noise_threshold_bits = noise_threshold_bits

        # Tracking state
        self._states: Dict[int, NoiseState] = {}  # ct_id -> state
        self._operation_log: List[Dict[str, Any]] = []
        self._warnings_issued = 0

        logger.info(
            f"NoiseTracker initialized: {initial_levels} levels, "
            f"scale=2^{scale_bits}, min_reserved={min_levels_required}"
        )

    def create_state(self, ct_id: int, level: int, scale: float) -> NoiseState:
        """
        Create initial state for a fresh ciphertext.

        Args:
            ct_id: Ciphertext identifier (object id)
            level: Current level
            scale: Current scale

        Returns:
            NoiseState for the ciphertext
        """
        state = NoiseState(
            level=level,
            scale=scale,
            estimated_noise_bits=0.0,
            operations=["encrypt"],
        )
        self._states[ct_id] = state
        return state

    def get_state(self, ct_id: int) -> Optional[NoiseState]:
        """Get current state for a ciphertext."""
        return self._states.get(ct_id)

    def update_state(
        self,
        ct_id: int,
        operation: str,
        new_level: Optional[int] = None,
        new_scale: Optional[float] = None,
    ) -> NoiseState:
        """
        Update state after an operation.

        Args:
            ct_id: Ciphertext identifier
            operation: Operation name (from OPERATION_COSTS)
            new_level: New level (if changed)
            new_scale: New scale (if changed)

        Returns:
            Updated NoiseState

        Raises:
            NoiseBudgetExhaustedError: If operation would exhaust budget
        """
        state = self._states.get(ct_id)
        if state is None:
            # Create default state
            state = NoiseState(
                level=self.initial_levels - 1,
                scale=2.0 ** self.scale_bits,
                estimated_noise_bits=0.0,
            )

        cost = OPERATION_COSTS.get(operation)
        if cost is None:
            logger.warning(f"Unknown operation: {operation}")
            cost = OperationCost(operation, 0, 1.0, "unknown")

        # Check if we have enough levels
        projected_level = (new_level if new_level is not None
                          else state.level - cost.levels_consumed)

        if projected_level < self.min_levels_required:
            raise NoiseBudgetExhaustedError(
                f"Operation '{operation}' would exhaust noise budget. "
                f"Current level: {state.level}, projected: {projected_level}, "
                f"minimum required: {self.min_levels_required}",
                remaining_levels=projected_level,
                required_levels=self.min_levels_required,
            )

        # Update state
        state.level = projected_level
        state.estimated_noise_bits += cost.estimated_noise_bits
        state.operations.append(operation)

        if new_scale is not None:
            state.scale = new_scale

        # Check noise threshold
        if state.estimated_noise_bits > self.noise_threshold_bits:
            self._warnings_issued += 1
            logger.warning(
                f"Ciphertext {ct_id} noise ({state.estimated_noise_bits:.1f} bits) "
                f"exceeds threshold ({self.noise_threshold_bits} bits)"
            )

        # Log operation
        self._operation_log.append({
            "ct_id": ct_id,
            "operation": operation,
            "level_before": state.level + cost.levels_consumed,
            "level_after": state.level,
            "noise_bits": state.estimated_noise_bits,
        })

        self._states[ct_id] = state
        return state

    def check_can_operate(
        self,
        ct_id: int,
        operations: List[str]
    ) -> bool:
        """
        Check if a sequence of operations can be performed.

        Args:
            ct_id: Ciphertext identifier
            operations: List of operation names

        Returns:
            True if operations are safe, False otherwise
        """
        state = self._states.get(ct_id)
        current_level = state.level if state else (self.initial_levels - 1)

        total_level_cost = sum(
            OPERATION_COSTS.get(op, OPERATION_COSTS["multiply_plain"]).levels_consumed
            for op in operations
        )

        return (current_level - total_level_cost) >= self.min_levels_required

    def estimate_lora_levels_needed(
        self,
        hidden_dim: int,
        rank: int,
        out_dim: int,
        scaling: bool = True
    ) -> int:
        """
        Estimate levels needed for LoRA delta computation.

        LoRA: delta = scaling * (x @ A^T @ B^T)

        With column packing (MOAI):
        - First matmul: multiply_plain + rescale = 1 level
        - Second matmul: multiply_plain + rescale = 1 level
        - Scaling (if needed): multiply_plain + rescale = 1 level

        Args:
            hidden_dim: Input dimension
            rank: LoRA rank
            out_dim: Output dimension
            scaling: Whether scaling will be applied

        Returns:
            Estimated levels needed
        """
        # Two matmuls with rescale
        levels = 2

        # Scaling adds one more level
        if scaling:
            levels += 1

        # Reserve minimum for decryption
        levels += self.min_levels_required

        return levels

    def get_metrics(self) -> Dict[str, Any]:
        """Get tracker metrics."""
        active_states = len(self._states)
        total_ops = len(self._operation_log)

        level_dist = {}
        for state in self._states.values():
            level_dist[state.level] = level_dist.get(state.level, 0) + 1

        return {
            "initial_levels": self.initial_levels,
            "min_levels_required": self.min_levels_required,
            "active_ciphertexts": active_states,
            "total_operations": total_ops,
            "warnings_issued": self._warnings_issued,
            "level_distribution": level_dist,
        }

    def get_operation_summary(self) -> Dict[str, int]:
        """Get summary of operations performed."""
        summary = {}
        for log in self._operation_log:
            op = log["operation"]
            summary[op] = summary.get(op, 0) + 1
        return summary

    def reset(self) -> None:
        """Reset tracker state."""
        self._states.clear()
        self._operation_log.clear()
        self._warnings_issued = 0


def create_tracker_from_context(context_params: Dict[str, Any]) -> NoiseTracker:
    """
    Create a NoiseTracker from CKKS context parameters.

    Args:
        context_params: Dict from backend.get_context_params()

    Returns:
        Configured NoiseTracker
    """
    return NoiseTracker(
        initial_levels=context_params.get("coeff_modulus_chain_length", 4),
        scale_bits=context_params.get("scale_bits", 40),
        min_levels_required=1,
    )
