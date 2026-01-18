"""
TensorGuard Edge Client Core
"""

import io
import logging
from typing import Optional, List, Dict, Any

import numpy as np
from ..schemas.common import Demonstration, ShieldConfig, ClientStatus
from ..utils.exceptions import ValidationError
from .adapters import VLAAdapter

logger = logging.getLogger(__name__)

class EdgeClient:
    """
    Primary interface for robot-side TensorGuard integration.
    """
    def __init__(self, config: Optional[ShieldConfig] = None):
        self.config = config or ShieldConfig()
        self.adapter: Optional[VLAAdapter] = None
        self.demonstrations: List[Demonstration] = []
        self.submissions_count = 0
        
    def set_adapter(self, adapter: VLAAdapter):
        """Bind a model adapter to this client."""
        self.adapter = adapter
        
    def add_demonstration(self, demo: Demonstration):
        """Queue a demonstration for the next training round."""
        self.demonstrations.append(demo)
        
    def get_status(self) -> ClientStatus:
        """Get current client status."""
        return ClientStatus(
            pending_submissions=len(self.demonstrations),
            total_submissions=self.submissions_count,
            privacy_budget_remaining=self.config.dp_epsilon,
            last_model_version="2.0.0",
            connection_status="online"
        )
        
    def process_round(self) -> bytes:
        """
        Compute gradients, apply privacy noise, and package for submission.
        """
        if not self.adapter:
            raise ValidationError("Adapter not set")
        
        if not self.demonstrations:
            return b""

        gradient_batches: List[Dict[str, np.ndarray]] = []
        for demo in self.demonstrations:
            gradients = self.adapter.compute_gradients(demo)
            if not gradients:
                raise ValidationError("Empty gradients received from adapter")
            gradient_batches.append(gradients)

        aggregated: Dict[str, np.ndarray] = {}
        for key in gradient_batches[0].keys():
            stacked = np.stack([g[key] for g in gradient_batches], axis=0)
            aggregated[key] = np.mean(stacked, axis=0)

        # Clip gradients to configured norm
        max_norm = float(self.config.max_gradient_norm)
        for key, value in aggregated.items():
            norm = np.linalg.norm(value)
            if norm > max_norm > 0:
                aggregated[key] = value * (max_norm / (norm + 1e-12))

        # Apply sparsity by keeping top-k magnitudes (deterministic)
        sparsity = float(self.config.sparsity)
        if 0 < sparsity < 1:
            for key, value in aggregated.items():
                flat = value.flatten()
                k = max(1, int(flat.size * (1 - sparsity)))
                if k < flat.size:
                    threshold = np.partition(np.abs(flat), -k)[-k]
                    mask = np.abs(value) >= threshold
                    aggregated[key] = value * mask

        # Apply DP noise calibrated by epsilon (basic Gaussian mechanism)
        epsilon = max(float(self.config.dp_epsilon), 1e-6)
        noise_scale = 1.0 / epsilon
        rng = np.random.default_rng()
        for key, value in aggregated.items():
            aggregated[key] = value + rng.normal(0, noise_scale, size=value.shape)

        buffer = io.BytesIO()
        np.savez_compressed(buffer, **aggregated)

        self.submissions_count += 1
        self.demonstrations = []
        return buffer.getvalue()

def create_client(model_type: str = "pi0", **kwargs) -> EdgeClient:
    """Factory function for creating an EdgeClient."""
    config = ShieldConfig(model_type=model_type, **kwargs)
    return EdgeClient(config)
