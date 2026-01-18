"""
Graceful Degradation Manager

Manages system behavior under resource constraints and failures,
enabling controlled degradation while maintaining core functionality.
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Set
import logging

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """System degradation levels."""
    NORMAL = "normal"           # Full functionality
    LIGHT = "light"             # Minor features disabled
    MODERATE = "moderate"       # Non-essential features disabled
    HEAVY = "heavy"             # Core features only
    CRITICAL = "critical"       # Emergency mode, minimal operation
    MAINTENANCE = "maintenance" # System under maintenance


@dataclass
class FeatureConfig:
    """Configuration for a degradable feature."""
    name: str
    description: str = ""
    min_level: DegradationLevel = DegradationLevel.NORMAL  # Level at which disabled
    priority: int = 50  # 0-100, higher = more important
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class DegradationTrigger:
    """Condition that triggers degradation."""
    name: str
    check_fn: Callable[[], bool]  # Returns True if degradation needed
    target_level: DegradationLevel
    message: str = ""
    cooldown_seconds: float = 60.0
    last_triggered: float = 0.0


@dataclass
class DegradationState:
    """Current degradation state."""
    level: DegradationLevel
    reason: str
    triggered_at: float
    triggered_by: str = ""
    disabled_features: Set[str] = field(default_factory=set)
    active_triggers: List[str] = field(default_factory=list)


class GracefulDegradationManager:
    """
    Manages graceful degradation of system features based on
    resource availability and system health.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._features: Dict[str, FeatureConfig] = {}
        self._triggers: Dict[str, DegradationTrigger] = {}
        self._current_state = DegradationState(
            level=DegradationLevel.NORMAL,
            reason="System operating normally",
            triggered_at=time.time()
        )
        self._state_history: List[DegradationState] = []
        self._listeners: List[Callable[[DegradationState, DegradationState], None]] = []
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._monitor_interval = 10.0

        # Level priority mapping (higher = more degraded)
        self._level_priority = {
            DegradationLevel.NORMAL: 0,
            DegradationLevel.LIGHT: 1,
            DegradationLevel.MODERATE: 2,
            DegradationLevel.HEAVY: 3,
            DegradationLevel.CRITICAL: 4,
            DegradationLevel.MAINTENANCE: 5,
        }

    def register_feature(self, config: FeatureConfig) -> "GracefulDegradationManager":
        """Register a degradable feature."""
        with self._lock:
            self._features[config.name] = config
        logger.info(f"Registered degradable feature: {config.name}")
        return self

    def register_trigger(self, trigger: DegradationTrigger) -> "GracefulDegradationManager":
        """Register a degradation trigger."""
        with self._lock:
            self._triggers[trigger.name] = trigger
        logger.info(f"Registered degradation trigger: {trigger.name}")
        return self

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled at current degradation level."""
        with self._lock:
            feature = self._features.get(feature_name)
            if not feature:
                return True  # Unknown features default to enabled

            if not feature.enabled:
                return False

            current_priority = self._level_priority[self._current_state.level]
            min_priority = self._level_priority[feature.min_level]

            # Feature disabled if current level >= feature's minimum level
            return current_priority < min_priority

    def get_enabled_features(self) -> List[str]:
        """Get list of currently enabled features."""
        with self._lock:
            return [
                name for name, feature in self._features.items()
                if self.is_feature_enabled(name)
            ]

    def get_disabled_features(self) -> List[str]:
        """Get list of currently disabled features."""
        with self._lock:
            return [
                name for name, feature in self._features.items()
                if not self.is_feature_enabled(name)
            ]

    @property
    def current_level(self) -> DegradationLevel:
        """Get current degradation level."""
        return self._current_state.level

    @property
    def current_state(self) -> DegradationState:
        """Get current degradation state."""
        with self._lock:
            return self._current_state

    def set_level(
        self,
        level: DegradationLevel,
        reason: str,
        triggered_by: str = "manual"
    ) -> bool:
        """Manually set degradation level."""
        return self._transition_to(level, reason, triggered_by)

    def _transition_to(
        self,
        level: DegradationLevel,
        reason: str,
        triggered_by: str
    ) -> bool:
        """Transition to a new degradation level."""
        with self._lock:
            old_state = self._current_state

            if old_state.level == level:
                return False

            # Calculate disabled features at new level
            disabled = set()
            for name, feature in self._features.items():
                current_priority = self._level_priority[level]
                min_priority = self._level_priority[feature.min_level]
                if current_priority >= min_priority:
                    disabled.add(name)

            new_state = DegradationState(
                level=level,
                reason=reason,
                triggered_at=time.time(),
                triggered_by=triggered_by,
                disabled_features=disabled,
                active_triggers=list(self._triggers.keys())
            )

            self._current_state = new_state
            self._state_history.append(old_state)

            # Keep only last 100 state changes
            if len(self._state_history) > 100:
                self._state_history = self._state_history[-100:]

            logger.info(
                f"Degradation level changed: {old_state.level.value} -> {level.value} "
                f"(reason: {reason}, by: {triggered_by})"
            )

            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(old_state, new_state)
                except Exception as e:
                    logger.error(f"Degradation listener error: {e}")

            return True

    def add_listener(
        self,
        listener: Callable[[DegradationState, DegradationState], None]
    ):
        """Add a listener for state changes."""
        self._listeners.append(listener)

    def check_triggers(self) -> DegradationLevel:
        """Check all triggers and return highest triggered level."""
        with self._lock:
            highest_level = DegradationLevel.NORMAL
            highest_trigger = None
            now = time.time()

            for name, trigger in self._triggers.items():
                # Check cooldown
                if now - trigger.last_triggered < trigger.cooldown_seconds:
                    continue

                try:
                    if trigger.check_fn():
                        trigger_priority = self._level_priority[trigger.target_level]
                        current_highest = self._level_priority[highest_level]

                        if trigger_priority > current_highest:
                            highest_level = trigger.target_level
                            highest_trigger = trigger

                except Exception as e:
                    logger.error(f"Trigger check error for '{name}': {e}")

            if highest_trigger and highest_level != self._current_state.level:
                highest_trigger.last_triggered = now
                self._transition_to(
                    highest_level,
                    highest_trigger.message or f"Triggered by {highest_trigger.name}",
                    highest_trigger.name
                )

            return highest_level

    def recover_to_normal(self) -> bool:
        """Attempt to recover to normal operation."""
        with self._lock:
            # Check if all triggers are clear
            for name, trigger in self._triggers.items():
                try:
                    if trigger.check_fn():
                        logger.info(f"Cannot recover: trigger '{name}' still active")
                        return False
                except Exception as e:
                    logger.error(f"Trigger check error during recovery: {e}")
                    return False

            if self._current_state.level != DegradationLevel.NORMAL:
                return self._transition_to(
                    DegradationLevel.NORMAL,
                    "All triggers cleared, recovering to normal operation",
                    "auto_recovery"
                )

            return True

    def start_monitoring(self, interval: float = 10.0):
        """Start background monitoring of triggers."""
        if self._running:
            return

        self._monitor_interval = interval
        self._running = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="degradation-monitor"
        )
        self._monitor_thread.start()
        logger.info("Degradation monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Degradation monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self.check_triggers()

                # Attempt recovery if degraded and conditions allow
                if self._current_state.level != DegradationLevel.NORMAL:
                    self.recover_to_normal()

            except Exception as e:
                logger.error(f"Degradation monitoring error: {e}")

            if self._stop_event.wait(timeout=self._monitor_interval):
                break

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive degradation status."""
        with self._lock:
            return {
                "level": self._current_state.level.value,
                "reason": self._current_state.reason,
                "triggered_at": self._current_state.triggered_at,
                "triggered_by": self._current_state.triggered_by,
                "enabled_features": self.get_enabled_features(),
                "disabled_features": list(self._current_state.disabled_features),
                "registered_features": len(self._features),
                "registered_triggers": len(self._triggers),
                "active_triggers": [
                    name for name, t in self._triggers.items()
                    if time.time() - t.last_triggered < t.cooldown_seconds
                ],
            }

    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get state change history."""
        with self._lock:
            return [
                {
                    "level": state.level.value,
                    "reason": state.reason,
                    "triggered_at": state.triggered_at,
                    "triggered_by": state.triggered_by,
                }
                for state in self._state_history
            ]


# Global degradation manager instance
degradation_manager = GracefulDegradationManager()


def degradable_feature(
    feature_name: str,
    fallback_fn: Optional[Callable] = None,
    min_level: DegradationLevel = DegradationLevel.MODERATE
):
    """
    Decorator for degradable feature functions.

    Usage:
        @degradable_feature("advanced_analytics", fallback_fn=simple_analytics)
        def advanced_analytics():
            ...
    """
    def decorator(func: Callable) -> Callable:
        # Register the feature
        degradation_manager.register_feature(FeatureConfig(
            name=feature_name,
            description=f"Function: {func.__name__}",
            min_level=min_level
        ))

        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if degradation_manager.is_feature_enabled(feature_name):
                return func(*args, **kwargs)
            elif fallback_fn:
                logger.debug(f"Feature '{feature_name}' degraded, using fallback")
                return fallback_fn(*args, **kwargs)
            else:
                logger.debug(f"Feature '{feature_name}' disabled, returning None")
                return None

        return wrapper
    return decorator


# Default system resource triggers
def setup_default_triggers():
    """Setup default resource-based degradation triggers."""
    try:
        import psutil

        def check_high_cpu():
            return psutil.cpu_percent(interval=0.1) > 90

        def check_high_memory():
            return psutil.virtual_memory().percent > 90

        def check_critical_memory():
            return psutil.virtual_memory().percent > 95

        def check_disk_full():
            return psutil.disk_usage('/').percent > 95

        degradation_manager.register_trigger(DegradationTrigger(
            name="high_cpu",
            check_fn=check_high_cpu,
            target_level=DegradationLevel.LIGHT,
            message="High CPU usage detected",
            cooldown_seconds=30.0
        ))

        degradation_manager.register_trigger(DegradationTrigger(
            name="high_memory",
            check_fn=check_high_memory,
            target_level=DegradationLevel.MODERATE,
            message="High memory usage detected",
            cooldown_seconds=30.0
        ))

        degradation_manager.register_trigger(DegradationTrigger(
            name="critical_memory",
            check_fn=check_critical_memory,
            target_level=DegradationLevel.HEAVY,
            message="Critical memory pressure",
            cooldown_seconds=10.0
        ))

        degradation_manager.register_trigger(DegradationTrigger(
            name="disk_full",
            check_fn=check_disk_full,
            target_level=DegradationLevel.CRITICAL,
            message="Disk space critically low",
            cooldown_seconds=60.0
        ))

        logger.info("Default degradation triggers configured")

    except ImportError:
        logger.warning("psutil not available, default triggers not configured")
