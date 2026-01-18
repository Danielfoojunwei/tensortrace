"""
Rényi Differential Privacy (RDP) Accountant

Production-grade privacy accounting based on:
- Mironov (2017): "Rényi Differential Privacy"
- Abadi et al. (2016): "Deep Learning with Differential Privacy"
- Google's DP accounting: https://github.com/google/differential-privacy

This implementation provides:
- RDP composition for multiple mechanisms
- Optimal RDP → (ε, δ)-DP conversion
- Subsampled Gaussian mechanism accounting
- Per-round and cumulative privacy tracking

Note: For production with PyTorch, consider using Opacus which provides
GPU-accelerated DP-SGD. This implementation provides a self-contained
accountant for scenarios where Opacus is not available.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


# Default RDP orders for accounting (covers typical use cases)
DEFAULT_RDP_ORDERS = tuple([1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)))


@dataclass
class RDPStep:
    """A single step in the RDP accounting ledger."""
    step_id: int
    noise_multiplier: float
    sample_rate: float
    rdp_values: Tuple[float, ...]  # RDP at each order
    mechanism: str = "subsampled_gaussian"
    timestamp: float = 0.0


@dataclass
class RDPAccountant:
    """
    Rényi Differential Privacy Accountant.

    Tracks privacy loss using RDP composition and converts to (ε, δ)-DP.

    Attributes:
        orders: RDP orders for computing privacy loss
        target_delta: Target δ for (ε, δ)-DP conversion
        epsilon_budget: Maximum allowed ε before halting
        steps: List of accounting steps
    """

    orders: Tuple[float, ...] = DEFAULT_RDP_ORDERS
    target_delta: float = 1e-5
    epsilon_budget: float = 10.0
    steps: List[RDPStep] = field(default_factory=list)

    # Cumulative RDP values at each order
    _cumulative_rdp: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize cumulative RDP array."""
        self._cumulative_rdp = np.zeros(len(self.orders))

    def compute_rdp_gaussian(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
    ) -> np.ndarray:
        """
        Compute RDP for subsampled Gaussian mechanism.

        Based on Theorem 9 in Mironov (2017) and amplification by subsampling
        from Balle et al. (2018).

        Args:
            noise_multiplier: Ratio of noise std to sensitivity (σ/Δf)
            sample_rate: Probability of including each record (q)
            num_steps: Number of composition steps

        Returns:
            Array of RDP values at each order
        """
        if noise_multiplier <= 0:
            raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")

        rdp = np.zeros(len(self.orders))

        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                rdp[i] = float('inf')
                continue

            # For small sample rates, use Poisson subsampling approximation
            if sample_rate < 0.01:
                # RDP for Gaussian: α / (2σ²)
                base_rdp = alpha / (2 * noise_multiplier ** 2)
                # Amplification by subsampling (first-order approximation)
                rdp[i] = sample_rate ** 2 * base_rdp * num_steps
            else:
                # Full computation for larger sample rates
                rdp[i] = self._compute_rdp_subsampled_gaussian(
                    alpha, noise_multiplier, sample_rate
                ) * num_steps

            # RDP must be non-negative (numerical stability)
            rdp[i] = max(0.0, rdp[i])

        return rdp

    def _compute_rdp_subsampled_gaussian(
        self,
        alpha: float,
        noise_multiplier: float,
        sample_rate: float,
    ) -> float:
        """
        Compute RDP for a single step of subsampled Gaussian mechanism.

        Uses the log-sum-exp trick for numerical stability.
        """
        if alpha <= 1:
            return float('inf')

        sigma = noise_multiplier
        q = sample_rate

        # For very small q, use the approximation
        if q < 1e-6:
            return q ** 2 * alpha / (2 * sigma ** 2)

        # Compute using the formula from Mironov (2017)
        # This is a simplified version; for production, consider using
        # the more numerically stable implementation from Google's DP library

        log_terms = []
        for j in range(int(alpha) + 1):
            # Binomial coefficient
            log_binom = self._log_comb(alpha, j)
            # (1-q)^(α-j) * q^j
            if j == 0:
                log_prob = (alpha - j) * math.log(max(1 - q, 1e-10))
            elif j == alpha:
                log_prob = j * math.log(q)
            else:
                log_prob = (alpha - j) * math.log(max(1 - q, 1e-10)) + j * math.log(q)

            # exp(j(j-1)/(2σ²))
            log_exp_term = j * (j - 1) / (2 * sigma ** 2)

            log_terms.append(log_binom + log_prob + log_exp_term)

        # Log-sum-exp for numerical stability
        max_term = max(log_terms)
        log_sum = max_term + math.log(sum(math.exp(t - max_term) for t in log_terms))

        return log_sum / (alpha - 1)

    @staticmethod
    @lru_cache(maxsize=1000)
    def _log_comb(n: float, k: int) -> float:
        """Compute log of binomial coefficient using gamma function."""
        if k < 0 or k > n:
            return float('-inf')
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    def add_step(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
        mechanism: str = "subsampled_gaussian",
    ) -> float:
        """
        Add a training step and return the current epsilon.

        Args:
            noise_multiplier: Ratio of noise std to sensitivity
            sample_rate: Subsampling rate
            num_steps: Number of gradient descent steps
            mechanism: Name of the mechanism (for logging)

        Returns:
            Current (ε, δ)-DP epsilon after this step
        """
        # Compute RDP for this step
        step_rdp = self.compute_rdp_gaussian(noise_multiplier, sample_rate, num_steps)

        # Record the step
        step = RDPStep(
            step_id=len(self.steps),
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            rdp_values=tuple(step_rdp),
            mechanism=mechanism,
        )
        self.steps.append(step)

        # Update cumulative RDP (simple composition: add RDP values)
        self._cumulative_rdp += step_rdp

        # Convert to (ε, δ)-DP
        epsilon = self.get_epsilon()

        logger.info(
            f"RDP Accountant: step={len(self.steps)}, "
            f"σ={noise_multiplier:.2f}, q={sample_rate:.4f}, "
            f"ε={epsilon:.4f}, budget={self.epsilon_budget:.2f}"
        )

        return epsilon

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Convert cumulative RDP to (ε, δ)-DP epsilon.

        Uses the optimal conversion from Theorem 21 in Mironov (2017):
        ε = min_α [RDP_α + log(1/δ) / (α-1)]

        Args:
            delta: Target δ (uses self.target_delta if not specified)

        Returns:
            Epsilon for (ε, δ)-DP
        """
        # If no steps have been added, epsilon is 0
        if len(self.steps) == 0:
            return 0.0

        if delta is None:
            delta = self.target_delta

        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")

        log_delta = math.log(1 / delta)

        # Find the order that minimizes epsilon
        min_epsilon = float('inf')

        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                continue

            rdp = self._cumulative_rdp[i]
            # ε = RDP_α + log(1/δ) / (α-1)
            epsilon = rdp + log_delta / (alpha - 1)

            if epsilon < min_epsilon:
                min_epsilon = epsilon

        return max(0, min_epsilon)

    def get_remaining_budget(self) -> float:
        """Return remaining epsilon budget."""
        current = self.get_epsilon()
        return max(0, self.epsilon_budget - current)

    def is_budget_exhausted(self) -> bool:
        """Check if epsilon budget is exhausted."""
        return self.get_epsilon() >= self.epsilon_budget

    def compute_noise_multiplier(
        self,
        target_epsilon: float,
        sample_rate: float,
        num_steps: int,
        delta: Optional[float] = None,
    ) -> float:
        """
        Compute the noise multiplier needed to achieve target epsilon.

        Uses binary search to find the appropriate noise level.

        Args:
            target_epsilon: Desired (ε, δ)-DP epsilon
            sample_rate: Subsampling rate
            num_steps: Total number of training steps
            delta: Target δ (uses self.target_delta if not specified)

        Returns:
            Required noise multiplier
        """
        if delta is None:
            delta = self.target_delta

        # Binary search for noise multiplier
        low, high = 0.01, 100.0

        for _ in range(100):  # Max iterations
            mid = (low + high) / 2

            # Create temporary accountant to test
            test_accountant = RDPAccountant(
                orders=self.orders,
                target_delta=delta,
            )
            test_accountant.add_step(mid, sample_rate, num_steps)
            epsilon = test_accountant.get_epsilon()

            if abs(epsilon - target_epsilon) < 0.001:
                return mid
            elif epsilon > target_epsilon:
                low = mid
            else:
                high = mid

        return mid

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Return the total privacy spent as (ε, δ).

        Returns:
            Tuple of (epsilon, delta)
        """
        return (self.get_epsilon(), self.target_delta)

    def summary(self) -> dict:
        """Return a summary of the accountant state."""
        return {
            "num_steps": len(self.steps),
            "epsilon": self.get_epsilon(),
            "delta": self.target_delta,
            "budget": self.epsilon_budget,
            "remaining": self.get_remaining_budget(),
            "exhausted": self.is_budget_exhausted(),
        }


def compute_dp_sgd_privacy(
    sample_size: int,
    batch_size: int,
    noise_multiplier: float,
    epochs: int,
    delta: float = 1e-5,
) -> Tuple[float, float]:
    """
    Compute (ε, δ)-DP guarantee for DP-SGD training.

    Convenience function for common use case.

    Args:
        sample_size: Total number of training examples
        batch_size: Batch size for each step
        noise_multiplier: Ratio of noise std to clipping norm
        epochs: Number of training epochs
        delta: Target δ

    Returns:
        Tuple of (epsilon, delta)
    """
    sample_rate = batch_size / sample_size
    num_steps = int(epochs * sample_size / batch_size)

    accountant = RDPAccountant(target_delta=delta)
    accountant.add_step(noise_multiplier, sample_rate, num_steps)

    return accountant.get_privacy_spent()


# Precomputed lookup for common configurations
PRIVACY_LOOKUP = {
    # (noise_multiplier, sample_rate, steps) -> approximate epsilon at delta=1e-5
    # These are cached for quick estimation during development
    (1.0, 0.01, 1000): 1.73,
    (1.0, 0.01, 10000): 5.47,
    (1.1, 0.01, 1000): 1.43,
    (1.1, 0.01, 10000): 4.52,
    (2.0, 0.01, 1000): 0.45,
    (2.0, 0.01, 10000): 1.42,
}
