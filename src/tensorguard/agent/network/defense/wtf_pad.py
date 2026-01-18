"""
WTF-PAD - Zero-Delay Adaptive Padding Defense

Implements adaptive padding based on Juarez et al. (ESORICS 2016).
"""

import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Awaitable
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class WTFPADState(Enum):
    """WTF-PAD state machine states."""
    GAP = "gap"
    BURST = "burst"


@dataclass
class IATHistogram:
    """Histogram of inter-arrival times for sampling."""
    bins: np.ndarray
    counts: np.ndarray
    
    @classmethod
    def from_samples(cls, iats: np.ndarray, n_bins: int = 100) -> "IATHistogram":
        if len(iats) == 0:
            bins = np.linspace(0.001, 0.1, n_bins + 1)
            counts = np.ones(n_bins)
        else:
            counts, bins = np.histogram(iats, bins=n_bins)
            counts = np.maximum(counts, 1)
        return cls(bins=bins, counts=counts)
    
    def sample(self, rng: np.random.Generator) -> float:
        probs = self.counts / self.counts.sum()
        bin_idx = rng.choice(len(self.counts), p=probs)
        iat = rng.uniform(self.bins[bin_idx], self.bins[bin_idx + 1])
        return float(iat)


@dataclass
class WTFPADConfig:
    """WTF-PAD configuration parameters."""
    burst_histogram_bins: int = 100
    gap_histogram_bins: int = 100
    min_dummy_size: int = 64
    max_dummy_size: int = 1400
    min_iat_s: float = 0.001
    max_iat_s: float = 0.5
    gap_threshold_s: float = 0.1


class WTFPAD:
    """Zero-delay Adaptive Padding Defense."""
    
    def __init__(
        self,
        config: Optional[WTFPADConfig] = None,
        burst_histogram: Optional[IATHistogram] = None,
        gap_histogram: Optional[IATHistogram] = None,
        random_seed: int = 42
    ):
        self.config = config or WTFPADConfig()
        self.rng = np.random.default_rng(random_seed)
        
        self._burst_histogram = burst_histogram
        self._gap_histogram = gap_histogram
        
        self.state = WTFPADState.GAP
        self.last_packet_time = 0.0
        self.target_iat: Optional[float] = None
        self._timer_task: Optional[asyncio.Task] = None
        self._send_callback: Optional[Callable[[bytes], Awaitable[None]]] = None
        
        self.stats = {
            "real_packets": 0,
            "dummy_packets": 0,
            "state_transitions": 0,
        }
    
    def _get_histogram(self) -> IATHistogram:
        if self.state == WTFPADState.BURST:
            if self._burst_histogram is None:
                return IATHistogram.from_samples(
                    np.array([0.01, 0.02, 0.03, 0.05]),
                    self.config.burst_histogram_bins
                )
            return self._burst_histogram
        else:
            if self._gap_histogram is None:
                return IATHistogram.from_samples(
                    np.array([0.1, 0.2, 0.3, 0.5]),
                    self.config.gap_histogram_bins
                )
            return self._gap_histogram
    
    def _sample_target_iat(self) -> float:
        histogram = self._get_histogram()
        iat = histogram.sample(self.rng)
        return max(self.config.min_iat_s, min(iat, self.config.max_iat_s))
    
    def _generate_dummy_packet(self) -> bytes:
        size = self.rng.integers(self.config.min_dummy_size, self.config.max_dummy_size + 1)
        return b'\x00' * size
    
    async def _timeout_handler(self) -> None:
        try:
            while True:
                if self.target_iat is None:
                    self.target_iat = self._sample_target_iat()
                
                await asyncio.sleep(self.target_iat)
                
                if self._send_callback:
                    dummy = self._generate_dummy_packet()
                    await self._send_callback(dummy)
                    self.stats["dummy_packets"] += 1
                
                self.target_iat = self._sample_target_iat()
                
        except asyncio.CancelledError:
            pass
    
    async def on_packet(self, packet: bytes) -> bytes:
        current_time = time.monotonic()
        iat = current_time - self.last_packet_time if self.last_packet_time > 0 else 0.0
        
        if iat > self.config.gap_threshold_s:
            self.state = WTFPADState.GAP
        else:
            self.state = WTFPADState.BURST
        
        self.last_packet_time = current_time
        self.target_iat = self._sample_target_iat()
        self.stats["real_packets"] += 1
        
        return packet
    
    async def start(self, send_callback: Callable[[bytes], Awaitable[None]]) -> None:
        self._send_callback = send_callback
        self._timer_task = asyncio.create_task(self._timeout_handler())
        logger.info("WTF-PAD defense started")
    
    async def stop(self) -> None:
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass


@dataclass
class JitterConfig:
    """Configuration for timing obfuscation."""
    enabled: bool = True
    min_latency_ms: float = 5.0
    max_latency_ms: float = 50.0
    
class JitterBuffer:
    """
    Timing obfuscation buffer.
    Adds random delays to decorrelate packet arrival times from processing times.
    """
    def __init__(self, config: JitterConfig = JitterConfig()):
        self.config = config
        self.queue = asyncio.PriorityQueue()
        self._process_task: Optional[asyncio.Task] = None
        self._send_callback: Optional[Callable[[bytes], Awaitable[None]]] = None
        self.rng = np.random.default_rng()

    def start(self, send_callback: Callable[[bytes], Awaitable[None]]):
        self._send_callback = send_callback
        self._process_task = asyncio.create_task(self._process_queue())

    async def stop(self):
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

    async def send(self, data: bytes):
        """Schedule a packet for sending with random delay."""
        if not self.config.enabled:
            if self._send_callback:
                await self._send_callback(data)
            return

        delay_s = self.rng.uniform(self.config.min_latency_ms, self.config.max_latency_ms) / 1000.0
        send_time = time.monotonic() + delay_s
        await self.queue.put((send_time, data))

    async def _process_queue(self):
        while True:
            try:
                # Get next scheduled packet
                send_time, data = await self.queue.get()
                
                # Wait until scheduled time
                now = time.monotonic()
                if send_time > now:
                    await asyncio.sleep(send_time - now)
                
                if self._send_callback:
                    await self._send_callback(data)
                    
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Jitter buffer error: {e}")
