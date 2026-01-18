"""
Determinism Guard - Safety Constraints for RTPL

Ensures that traffic privacy defenses (FRONT, WTF-PAD) do not exceed 
the real-time latency and jitter requirements of the robot.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class DeterminismProfile:
    name: str
    max_latency_ms: float
    max_jitter_ms: float

PROFILES = {
    "surgical": DeterminismProfile("surgical", 0.5, 0.1),
    "collaborative": DeterminismProfile("collaborative", 2.0, 1.0),
    "warehouse": DeterminismProfile("warehouse", 10.0, 5.0),
    "lab": DeterminismProfile("lab", 50.0, 20.0),
}

@dataclass
class GuardResult:
    passed: bool
    violation_type: Optional[str] = None
    value: float = 0.0

class DeterminismGuard:
    """
    Monitors network performance and enforces safety bounds.
    """
    def __init__(self, max_added_latency_ms: float, max_jitter_ms: float):
        self.max_latency = max_added_latency_ms
        self.max_jitter = max_jitter_ms
        
    @classmethod
    def from_profile(cls, profile_name: str) -> "DeterminismGuard":
        profile = PROFILES.get(profile_name, PROFILES["collaborative"])
        return cls(profile.max_latency_ms, profile.max_jitter_ms)

    def check(self, measured_latency_ms: float, measured_jitter_ms: float) -> GuardResult:
        """Verify that current performance is within bounds."""
        if measured_latency_ms > self.max_latency:
            return GuardResult(False, "latency_exceeded", measured_latency_ms)
            
        if measured_jitter_ms > self.max_jitter:
            return GuardResult(False, "jitter_exceeded", measured_jitter_ms)
            
        return GuardResult(True)
