"""
TensorGuard Privacy Pipeline Components
Implementing Differential Privacy, Sparsification, and Compression.
"""

import numpy as np
import gzip
import msgpack  # Safe serialization (no RCE risk unlike pickle)
from typing import Dict, Any, Optional
from ..utils.logging import get_logger
from ..utils.exceptions import QualityWarning, ValidationError

logger = get_logger(__name__)

class GradientClipper:
    """Clips gradients to bounded norm for differential privacy."""
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def clip(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))
        clip_coef = min(self.max_norm / (total_norm + 1e-6), 1.0)
        return {k: v * clip_coef for k, v in gradients.items()}

class ExpertGater:
    """
    Expert-Driven Gating (v2.0).
    Based on instruction relevance (IOSP) instead of raw magnitude.
    Addresses parameter interference in heterogeneous robot fleets.
    """
    def __init__(self, gate_threshold: float = 0.1):
        self.gate_threshold = gate_threshold

    def gate(self, expert_grads: Dict[str, Dict[str, np.ndarray]], gate_weights: Dict[str, float]) -> Dict[str, np.ndarray]:
        combined = {}
        if not expert_grads or not gate_weights:
            return combined
            
        for expert, grads in expert_grads.items():
            weight = gate_weights.get(expert, 0.0)
            if weight > self.gate_threshold:
                for k, v in grads.items():
                    combined[k] = combined.get(k, 0) + v
        return combined

class RandomSparsifier:
    """
    Random Sparsification (Rand-K).
    Selects a random subset of gradients to transmit, ensuring data independence.
    
    Advantages (Miao et al., FedVLA):
    1. Privacy: Indices are chosen randomly, leaking no information about data distribution.
    2. Unbiased: Does not systematically ignore small updates (preventing 'gradient starvation').
    3. Robustness: Data-agnostic selection works better for heterogeneous fleets.
    """
    def __init__(self, sparsity_ratio: float = 0.01):
        """
        Args:
            sparsity_ratio (float): Fraction of parameters to keep (0.0 < ratio <= 1.0).
                                    e.g., 0.01 means keep 1% of parameters.
        """
        if not (0.0 < sparsity_ratio <= 1.0):
            raise ValueError(f"Sparsity ratio must be between 0 and 1, gave {sparsity_ratio}")
        self.sparsity_ratio = sparsity_ratio

    def sparsify(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result = {}
        for name, grad in gradients.items():
            flat = grad.flatten()
            num_elements = flat.size
            if num_elements == 0:
                result[name] = grad
                continue
                
            k = max(1, int(num_elements * self.sparsity_ratio))
            
            # Randomly select k indices
            # We use the CSPRNG-seeded generator implicitly via numpy if available,
            # but for indices selection, standard np.random is acceptable as the *indices*
            # themselves are the public pattern, and we want them to be uniform.
            # However, for strict consistency, better to be explicit or use standard choice.
            indices = np.random.choice(num_elements, size=k, replace=False)
            
            # Create sparse mask
            mask = np.zeros_like(flat, dtype=bool)
            mask[indices] = True
            
            # Apply mask
            sparse_flat = np.zeros_like(flat)
            sparse_flat[mask] = flat[mask]
            
            # Reshape back to original
            result[name] = sparse_flat.reshape(grad.shape)
            
        return result

class APHECompressor:
    """
    Neural compression using quantization.
    Uses msgpack for safe serialization (no RCE vulnerability).
    """
    def __init__(self, compression_ratio: int = 32):
        self.compression_ratio = compression_ratio
        self.bits = max(2, 32 // compression_ratio)
    
    def compress(self, gradients: Dict[str, np.ndarray]) -> bytes:
        compressed_data = {}
        for name, grad in gradients.items():
            v_min, v_max = float(grad.min()), float(grad.max())
            if v_max - v_min > 1e-8:
                normalized = (grad - v_min) / (v_max - v_min)
                levels = 2 ** self.bits
                quantized = np.round(normalized * (levels - 1)).astype(np.uint8)
            else:
                quantized = np.zeros_like(grad, dtype=np.uint8)
                v_min = v_max = 0.0
            
            compressed_data[name] = {
                'q': quantized.tobytes(),
                'dtype': str(quantized.dtype),
                'min': v_min,
                'max': v_max,
                'shape': list(grad.shape)
            }
        return gzip.compress(msgpack.packb(compressed_data, use_bin_type=True))
    
    def decompress(self, data: bytes) -> Dict[str, np.ndarray]:
        compressed_data = msgpack.unpackb(gzip.decompress(data), raw=False)
        result = {}
        levels = 2 ** self.bits
        for name, payload in compressed_data.items():
            quantized = np.frombuffer(payload['q'], dtype=np.uint8).reshape(payload['shape'])
            dequantized = quantized.astype(np.float32) / (levels - 1)
            result[name] = (dequantized * (payload['max'] - payload['min']) + payload['min'])
        return result


class QualityMonitor:
    """Monitors reconstruction integrity."""
    def __init__(self, mse_threshold: float = 0.05):
        self.mse_threshold = mse_threshold
        
    def check_quality(self, original: Dict[str, np.ndarray], reconstructed: Dict[str, np.ndarray]) -> float:
        max_mse = 0.0
        for k in original:
            orig_norm = original[k] / (np.max(np.abs(original[k])) + 1e-9)
            recon_norm = reconstructed[k] / (np.max(np.abs(reconstructed[k])) + 1e-9)
            mse = np.mean((orig_norm - recon_norm) ** 2)
            max_mse = max(max_mse, mse)
            
        if max_mse > self.mse_threshold:
            logger.info(f"High reconstruction error (MSE={max_mse:.4f})", extra={"mse": max_mse})
        return max_mse
