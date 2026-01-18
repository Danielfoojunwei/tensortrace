"""
Recovery Strategy Module

Provides retry policies, fallback handlers, and recovery strategies
for graceful error handling and automatic recovery.

Supports deterministic jitter mode (TG_DETERMINISTIC=true) for reproducible
retry behavior in testing and debugging scenarios.
"""

import time
import random
import functools
import threading
import logging
import hashlib
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, List, Type, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def _is_deterministic_mode() -> bool:
    """Check if deterministic mode is enabled via environment."""
    return os.getenv("TG_DETERMINISTIC", "false").lower() == "true"


def _compute_deterministic_jitter(
    func_name: str,
    attempt: int,
    request_id: Optional[str] = None,
    jitter_factor: float = 0.25,
    base_delay: float = 1.0
) -> float:
    """
    Compute deterministic jitter using hash-based calculation.

    Given the same inputs, this function always produces the same jitter value.
    This is useful for reproducible testing and debugging.

    Args:
        func_name: Name of the function being retried
        attempt: Current retry attempt number
        request_id: Optional request ID for additional uniqueness
        jitter_factor: Maximum jitter as a fraction of base_delay (0.0-1.0)
        base_delay: Base delay to calculate jitter range from

    Returns:
        Deterministic jitter value in range [-jitter_range, +jitter_range]
    """
    # Create a deterministic hash from inputs
    hash_input = f"{func_name}:{attempt}:{request_id or 'no_request_id'}"
    hash_bytes = hashlib.sha256(hash_input.encode()).digest()

    # Convert first 8 bytes to a float between 0 and 1
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
    normalized = hash_int / (2**64 - 1)  # Normalize to [0, 1]

    # Scale to [-jitter_range, +jitter_range]
    jitter_range = base_delay * jitter_factor
    jitter = (normalized * 2 - 1) * jitter_range

    return jitter


class BackoffStrategy(Enum):
    """Backoff strategies for retry logic."""
    CONSTANT = "constant"           # Same delay every time
    LINEAR = "linear"               # Linearly increasing delay
    EXPONENTIAL = "exponential"     # Exponentially increasing delay
    EXPONENTIAL_JITTER = "exponential_jitter"  # Exponential with random jitter


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.25  # 25% random jitter
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = ()


class RetryPolicy:
    """
    Configurable retry policy with multiple backoff strategies.

    Supports various backoff strategies, jitter, and exception filtering.

    When TG_DETERMINISTIC=true, jitter is computed using a hash-based approach
    that produces the same delay for the same (func_name, attempt, request_id)
    combination, enabling reproducible retry behavior for testing.
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        func_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        self.config = config or RetryConfig()
        self._attempt = 0
        self._func_name = func_name or "unknown"
        self._request_id = request_id

    def set_context(self, func_name: str, request_id: Optional[str] = None) -> None:
        """
        Set context for deterministic jitter calculation.

        Args:
            func_name: Name of the function being retried
            request_id: Optional request ID for additional uniqueness
        """
        self._func_name = func_name
        self._request_id = request_id

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.

        When TG_DETERMINISTIC=true, uses hash-based deterministic jitter
        instead of random jitter for reproducibility.
        """
        cfg = self.config
        strategy = cfg.backoff_strategy

        if strategy == BackoffStrategy.CONSTANT:
            delay = cfg.base_delay_seconds
        elif strategy == BackoffStrategy.LINEAR:
            delay = cfg.base_delay_seconds * (attempt + 1)
        elif strategy == BackoffStrategy.EXPONENTIAL:
            delay = cfg.base_delay_seconds * (cfg.backoff_multiplier ** attempt)
        elif strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            base_delay = cfg.base_delay_seconds * (cfg.backoff_multiplier ** attempt)

            if _is_deterministic_mode():
                # Use deterministic jitter for reproducible retries
                jitter = _compute_deterministic_jitter(
                    func_name=self._func_name,
                    attempt=attempt,
                    request_id=self._request_id,
                    jitter_factor=cfg.jitter_factor,
                    base_delay=base_delay
                )
                delay = base_delay + jitter
                logger.debug(
                    f"Deterministic jitter for {self._func_name} attempt {attempt}: "
                    f"base={base_delay:.3f}s, jitter={jitter:.3f}s, total={delay:.3f}s"
                )
            else:
                # Use random jitter for production variance
                jitter_range = base_delay * cfg.jitter_factor
                delay = base_delay + random.uniform(-jitter_range, jitter_range)
        else:
            delay = cfg.base_delay_seconds

        return min(delay, cfg.max_delay_seconds)

    def should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False
        return isinstance(exception, self.config.retryable_exceptions)

    def execute(
        self,
        func: Callable,
        *args,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            on_retry: Callback for retry events (attempt, exception)
            request_id: Optional request ID for deterministic jitter calculation
            *args, **kwargs: Arguments passed to func

        Returns:
            Result of successful function execution

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None
        cfg = self.config

        # Set context for deterministic jitter calculation
        self._func_name = getattr(func, '__name__', 'anonymous')
        if request_id:
            self._request_id = request_id

        for attempt in range(cfg.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if not self.should_retry(e):
                    logger.debug(f"Non-retryable exception: {type(e).__name__}")
                    raise

                if attempt >= cfg.max_retries:
                    logger.warning(
                        f"All {cfg.max_retries} retries exhausted for {func.__name__}"
                    )
                    raise

                delay = self.calculate_delay(attempt)
                logger.info(
                    f"Retry {attempt + 1}/{cfg.max_retries} for {func.__name__} "
                    f"after {delay:.2f}s: {e}"
                )

                if on_retry:
                    try:
                        on_retry(attempt + 1, e)
                    except Exception as callback_error:
                        logger.error(f"Retry callback error: {callback_error}")

                time.sleep(delay)

        raise last_exception

    def wrap(self, func: Callable) -> Callable:
        """Decorator to wrap a function with retry logic."""
        # Set func_name at decoration time for deterministic jitter
        self._func_name = getattr(func, '__name__', 'anonymous')

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper


class FallbackHandler:
    """
    Manages fallback functions for graceful degradation.

    Provides a chain of fallback options when primary operations fail.
    """

    def __init__(self, name: str):
        self.name = name
        self._primary: Optional[Callable] = None
        self._fallbacks: List[Callable] = []
        self._lock = threading.Lock()
        self._stats = {
            "primary_calls": 0,
            "primary_successes": 0,
            "fallback_calls": 0,
            "fallback_successes": 0,
            "total_failures": 0,
        }

    def set_primary(self, func: Callable) -> "FallbackHandler":
        """Set the primary function."""
        self._primary = func
        return self

    def add_fallback(self, func: Callable) -> "FallbackHandler":
        """Add a fallback function (ordered by priority)."""
        self._fallbacks.append(func)
        return self

    def execute(self, *args, **kwargs) -> Any:
        """
        Execute with fallback chain.

        Tries primary function first, then each fallback in order.
        """
        if not self._primary:
            raise ValueError(f"No primary function set for {self.name}")

        # Try primary
        try:
            with self._lock:
                self._stats["primary_calls"] += 1

            result = self._primary(*args, **kwargs)

            with self._lock:
                self._stats["primary_successes"] += 1

            return result

        except Exception as primary_error:
            logger.warning(
                f"Primary function failed for {self.name}: {primary_error}"
            )

            # Try fallbacks in order
            for idx, fallback in enumerate(self._fallbacks):
                try:
                    with self._lock:
                        self._stats["fallback_calls"] += 1

                    result = fallback(*args, **kwargs)

                    with self._lock:
                        self._stats["fallback_successes"] += 1

                    logger.info(f"Fallback {idx + 1} succeeded for {self.name}")
                    return result

                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback {idx + 1} failed for {self.name}: {fallback_error}"
                    )
                    continue

            # All fallbacks exhausted
            with self._lock:
                self._stats["total_failures"] += 1

            logger.error(f"All fallbacks exhausted for {self.name}")
            raise primary_error

    def get_stats(self) -> Dict[str, int]:
        """Get fallback handler statistics."""
        with self._lock:
            return dict(self._stats)

    def wrap(self, func: Callable) -> Callable:
        """Decorator to use this handler."""
        self._primary = func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(*args, **kwargs)

        return wrapper


@dataclass
class RecoveryAction:
    """Definition of a recovery action."""
    name: str
    action: Callable[[], bool]
    timeout_seconds: float = 30.0
    retries: int = 3
    priority: int = 0  # Higher = more urgent


class RecoveryStrategy:
    """
    Comprehensive recovery strategy manager.

    Orchestrates recovery actions with prioritization and state tracking.
    """

    def __init__(self, name: str):
        self.name = name
        self._actions: Dict[str, RecoveryAction] = {}
        self._lock = threading.RLock()
        self._recovery_in_progress = False
        self._last_recovery_time: Optional[float] = None
        self._recovery_history: List[Dict[str, Any]] = []

    def register_action(self, action: RecoveryAction) -> "RecoveryStrategy":
        """Register a recovery action."""
        with self._lock:
            self._actions[action.name] = action
        logger.info(f"Registered recovery action: {action.name}")
        return self

    def unregister_action(self, name: str):
        """Unregister a recovery action."""
        with self._lock:
            if name in self._actions:
                del self._actions[name]

    def execute_recovery(
        self,
        specific_actions: Optional[List[str]] = None,
        on_progress: Optional[Callable[[str, bool], None]] = None
    ) -> Dict[str, bool]:
        """
        Execute recovery actions.

        Args:
            specific_actions: If provided, only run these actions
            on_progress: Callback (action_name, success)

        Returns:
            Dict mapping action names to success status
        """
        with self._lock:
            if self._recovery_in_progress:
                logger.warning("Recovery already in progress")
                return {}

            self._recovery_in_progress = True

        try:
            results = {}

            # Get actions to execute, sorted by priority
            with self._lock:
                if specific_actions:
                    actions = [
                        self._actions[name]
                        for name in specific_actions
                        if name in self._actions
                    ]
                else:
                    actions = list(self._actions.values())

            actions.sort(key=lambda a: -a.priority)  # Higher priority first

            logger.info(f"Starting recovery for {self.name} ({len(actions)} actions)")

            for action in actions:
                success = self._execute_single_action(action)
                results[action.name] = success

                if on_progress:
                    try:
                        on_progress(action.name, success)
                    except Exception as e:
                        logger.error(f"Progress callback error: {e}")

            # Record recovery attempt
            recovery_record = {
                "timestamp": time.time(),
                "results": results,
                "success_rate": sum(results.values()) / max(1, len(results)),
            }

            with self._lock:
                self._recovery_history.append(recovery_record)
                self._last_recovery_time = time.time()

                # Keep only last 100 recovery records
                if len(self._recovery_history) > 100:
                    self._recovery_history = self._recovery_history[-100:]

            return results

        finally:
            with self._lock:
                self._recovery_in_progress = False

    def _execute_single_action(self, action: RecoveryAction) -> bool:
        """Execute a single recovery action with retries."""
        for attempt in range(action.retries):
            try:
                logger.info(
                    f"Executing recovery action '{action.name}' "
                    f"(attempt {attempt + 1}/{action.retries})"
                )

                # Execute with timeout
                result = action.action()

                if result:
                    logger.info(f"Recovery action '{action.name}' succeeded")
                    return True

                logger.warning(f"Recovery action '{action.name}' returned False")

            except Exception as e:
                logger.error(
                    f"Recovery action '{action.name}' failed: {e}"
                )

            if attempt < action.retries - 1:
                time.sleep(1.0)  # Brief pause between retries

        return False

    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """Get recovery history."""
        with self._lock:
            return list(self._recovery_history)

    @property
    def is_recovering(self) -> bool:
        """Check if recovery is in progress."""
        return self._recovery_in_progress


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
    retryable_exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for adding retry logic to a function.

    Usage:
        @retry(max_retries=3, backoff=BackoffStrategy.EXPONENTIAL)
        def unreliable_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        config = RetryConfig(
            max_retries=max_retries,
            base_delay_seconds=base_delay,
            backoff_strategy=backoff,
            retryable_exceptions=retryable_exceptions
        )
        policy = RetryPolicy(config)
        return policy.wrap(func)
    return decorator


@contextmanager
def recovery_context(
    recovery_strategy: RecoveryStrategy,
    auto_recover: bool = True
):
    """
    Context manager for automatic recovery on failure.

    Usage:
        with recovery_context(my_strategy):
            do_something_risky()
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Error in recovery context: {e}")
        if auto_recover:
            logger.info("Attempting automatic recovery...")
            recovery_strategy.execute_recovery()
        raise
