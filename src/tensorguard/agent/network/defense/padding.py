"""
Padding Defense - Simple Size-Based Padding

Implements the basic padding defense from arXiv:2312.06802 Section 7.1.
Usage: Round up packet sizes to fixed bucket sizes.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import struct
import logging

logger = logging.getLogger(__name__)


@dataclass
class PaddingConfig:
    """Padding defense configuration."""
    bucket_bytes: int = 200  # Round to nearest multiple
    max_mtu: int = 1500      # Maximum packet size
    padding_byte: int = 0x00 # Byte used for padding


class PaddingOnly:
    """
    Simple size-based padding defense.
    """
    
    def __init__(self, config: Optional[PaddingConfig] = None):
        self.config = config or PaddingConfig()
        
        self.stats = {
            "packets_padded": 0,
            "bytes_original": 0,
            "bytes_padded": 0,
        }
    
    def _calculate_padded_size(self, original_size: int) -> int:
        """Calculate the padded size for a packet."""
        bucket = self.config.bucket_bytes
        padded = ((original_size + bucket - 1) // bucket) * bucket
        return min(padded, self.config.max_mtu)
    
    def pad(self, data: bytes) -> bytes:
        """Pad a packet to the next bucket boundary."""
        original_size = len(data)
        padded_size = self._calculate_padded_size(original_size)
        padding_needed = padded_size - original_size - 4  # 4 bytes for length header
        
        if padding_needed < 0:
            padded_size = self.config.max_mtu
            padding_needed = padded_size - original_size - 4
        
        length_header = struct.pack('>I', original_size)
        padding = bytes([self.config.padding_byte]) * max(0, padding_needed)
        
        self.stats["packets_padded"] += 1
        self.stats["bytes_original"] += original_size
        self.stats["bytes_padded"] += padded_size
        
        return length_header + data + padding
    
    def strip(self, padded_data: bytes) -> bytes:
        """Strip padding from a packet."""
        if len(padded_data) < 4:
            return padded_data
        
        original_size = struct.unpack('>I', padded_data[:4])[0]
        return padded_data[4:4 + original_size]
    
    def defend_trace(self, packets: List[Dict]) -> List[Dict]:
        """Apply padding defense to a trace (simulation mode)."""
        defended = []
        for pkt in packets:
            padded_size = self._calculate_padded_size(pkt["s"])
            defended.append({
                **pkt,
                "s": padded_size,
                "original_size": pkt["s"]
            })
        return defended
