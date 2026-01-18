"""
Circuit Breaker Pattern Implementation

Prevents cascade failures by monitoring call success/failure rates
and temporarily blocking calls to failing services.
"""

import time
import threading
import functools
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, calls pass through
    OPEN = "open"          # Failing, calls are blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_seconds: float = 30.0       # Time before attempting half-open
    window_seconds: float = 60.0        # Rolling window for failure counting
    excluded_exceptions: tuple = ()     # Exceptions that don't count as failures


@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0
    current_state: CircuitState = CircuitState.CLOSED


class CircuitBreaker:
    """
    Production-grade circuit breaker for protecting external calls.

    Usage:
        breaker = CircuitBreaker("aggregator")

        @breaker.protect
        def call_aggregator():
            ...

        # Or manual usage:
        with breaker:
            call_aggregator()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()
        self._failure_times: deque = deque()  # Rolling window of failure timestamps
        self._half_open_successes = 0
        self._opened_at: Optional[float] = None

        self.stats = CircuitStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state with automatic half-open transition."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is blocking calls."""
        return self.state == CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._opened_at is None:
            return False
        elapsed = time.time() - self._opened_at
        return elapsed >= self.config.timeout_seconds

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state with callback notification."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self.stats.state_changes += 1
        self.stats.current_state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._half_open_successes = 0
            logger.warning(f"Circuit '{self.name}' OPENED - blocking calls")
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_successes = 0
            logger.info(f"Circuit '{self.name}' HALF-OPEN - testing recovery")
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._failure_times.clear()
            logger.info(f"Circuit '{self.name}' CLOSED - normal operation")

        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    def _clean_old_failures(self):
        """Remove failures outside the rolling window."""
        cutoff = time.time() - self.config.window_seconds
        while self._failure_times and self._failure_times[0] < cutoff:
            self._failure_times.popleft()

    def _record_failure(self, exception: Exception):
        """Record a failure and potentially open the circuit."""
        with self._lock:
            # Check if exception should be excluded
            if isinstance(exception, self.config.excluded_exceptions):
                return

            now = time.time()
            self._failure_times.append(now)
            self._clean_old_failures()

            self.stats.failed_calls += 1
            self.stats.last_failure_time = now

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately reopens
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if len(self._failure_times) >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _record_success(self):
        """Record a success and potentially close the circuit."""
        with self._lock:
            now = time.time()
            self.stats.successful_calls += 1
            self.stats.last_success_time = now

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        state = self.state  # Triggers potential half-open transition
        if state == CircuitState.OPEN:
            self.stats.rejected_calls += 1
            return False
        return True

    def __enter__(self):
        """Context manager entry."""
        if not self.allow_request():
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")
        self.stats.total_calls += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self._record_failure(exc_val)
        else:
            self._record_success()
        return False  # Don't suppress exceptions

    def protect(self, func: Callable) -> Callable:
        """Decorator to protect a function with this circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_times.clear()
            self._half_open_successes = 0
            logger.info(f"Circuit '{self.name}' manually reset")

    def get_health(self) -> Dict[str, Any]:
        """Get circuit breaker health information."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "stats": {
                    "total_calls": self.stats.total_calls,
                    "successful_calls": self.stats.successful_calls,
                    "failed_calls": self.stats.failed_calls,
                    "rejected_calls": self.stats.rejected_calls,
                    "state_changes": self.stats.state_changes,
                },
                "failure_rate": (
                    self.stats.failed_calls / max(1, self.stats.total_calls)
                ),
                "recent_failures": len(self._failure_times),
            }


class CircuitOpenError(Exception):
    """Raised when attempting to use an open circuit."""
    pass


class CircuitBreakerRegistry:
    """
    Central registry for managing multiple circuit breakers.

    Provides a single point for monitoring and managing all circuits
    in the system.
    """

    _instance: Optional["CircuitBreakerRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for global registry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._breakers: Dict[str, CircuitBreaker] = {}
                    cls._instance._registry_lock = threading.RLock()
        return cls._instance

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable] = None
    ) -> CircuitBreaker:
        """Get an existing circuit breaker or create a new one."""
        with self._registry_lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    config=config,
                    on_state_change=on_state_change
                )
            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def list_all(self) -> List[str]:
        """List all registered circuit breaker names."""
        return list(self._breakers.keys())

    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all circuit breakers."""
        return {
            name: breaker.get_health()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers to closed state."""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_open_circuits(self) -> List[str]:
        """Get names of all currently open circuits."""
        return [
            name for name, breaker in self._breakers.items()
            if breaker.is_open
        ]


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout_seconds: float = 30.0,
    excluded_exceptions: tuple = ()
) -> Callable:
    """
    Decorator factory for easily protecting functions with circuit breakers.

    Usage:
        @circuit_breaker("my_service")
        def call_service():
            ...
    """
    def decorator(func: Callable) -> Callable:
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            timeout_seconds=timeout_seconds,
            excluded_exceptions=excluded_exceptions
        )
        breaker = circuit_registry.get_or_create(name, config)
        return breaker.protect(func)
    return decorator
