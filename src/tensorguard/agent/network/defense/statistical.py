"""
Statistical Padding Defense

Implements advanced padding strategies based on statistical distributions (Skellam, Laplacian)
to defeat traffic analysis classifiers that rely on packet size fingerprinting.
"""

import numpy as np
import logging
from typing import Optional, List, Dict
import struct

try:
    from scipy.stats import skellam, laplace
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

class StatisticalPadding:
    """
    Advanced padding defense using statistical distributions.
    """
    
    def __init__(self, distribution: str = "skellam", mu1: float = 10.0, mu2: float = 10.0, scale: float = 1.0, max_mtu: int = 1400):
        self.distribution = distribution
        self.mu1 = mu1 # Skellam parameter
        self.mu2 = mu2 # Skellam parameter
        self.scale = scale # Laplacian parameter
        self.max_mtu = max_mtu
        self.rng = np.random.default_rng()
        
        if not HAS_SCIPY:
            logger.warning("Scipy not found. Statistical padding falling back to uniform/simple logic.")

    def _get_padding_amount(self, original_size: int) -> int:
        """Calculate amount of padding bytes to add."""
        if not HAS_SCIPY:
            # Fallback: Pad to random multiple of 16
            return 16 - (original_size % 16)
            
        try:
            if self.distribution == "skellam":
                # Skellam: Difference of two Poissons. Used to mask traffic directionality/bursts.
                # We interpret the absolute value sample as additional padding variance
                sample = abs(skellam.rvs(self.mu1, self.mu2))
                return int(sample) * 8 # Granularity
                
            elif self.distribution == "laplacian":
                # Laplacian: Symmetric noise, common in Differential Privacy
                sample = abs(laplace.rvs(loc=0, scale=self.scale))
                return int(sample) * 16
                
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Padding sampling error: {e}")
            return 8

    def pad(self, data: bytes) -> bytes:
        """Apply statistical padding."""
        original_size = len(data)
        pad_amount = self._get_padding_amount(original_size)
        
        # Ensure we don't exceed MTU (simplified fragmentation logic for this layer)
        # In reality, we'd need to fragment, but here we just cap padding
        max_padding = self.max_mtu - original_size - 4
        if max_padding < 0:
            # Packet too big already, no padding, maybe error
            pad_amount = 0
        else:
            pad_amount = min(pad_amount, max_padding)
            
        length_header = struct.pack('>I', original_size)
        padding = b'\x00' * pad_amount
        
        return length_header + data + padding

    def strip(self, padded_data: bytes) -> bytes:
        """Remove padding."""
        if len(padded_data) < 4:
            return padded_data
            
        original_size = struct.unpack('>I', padded_data[:4])[0]
        if original_size > len(padded_data) - 4:
            # Malformed or fragmented
            return padded_data[4:]
            
        return padded_data[4:4+original_size]
