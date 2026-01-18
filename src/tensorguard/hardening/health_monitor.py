"""
Health Monitoring System

Comprehensive health monitoring for all TensorGuard components with
hierarchical status aggregation and degradation detection.
"""

import time
import threading
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Any, Set
from collections import deque
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"           # Fully operational
    DEGRADED = "degraded"         # Operating with reduced capability
    UNHEALTHY = "unhealthy"       # Not functioning properly
    UNKNOWN = "unknown"           # Unable to determine status
    INITIALIZING = "initializing" # Component starting up


@dataclass
class ComponentHealth:
    """Health information for a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    error_count: int = 0
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "dependencies": self.dependencies,
            "error_count": self.error_count,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class HealthCheckConfig:
    """Configuration for health check behavior."""
    check_interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    failure_threshold: int = 3          # Failures before unhealthy
    degraded_threshold: int = 1         # Failures before degraded
    history_size: int = 100             # Number of results to keep


class HealthCheck:
    """Individual health check for a component."""

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], bool],
        config: Optional[HealthCheckConfig] = None,
        dependencies: Optional[List[str]] = None
    ):
        self.name = name
        self.check_fn = check_fn
        self.config = config or HealthCheckConfig()
        self.dependencies = dependencies or []

        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=self.config.history_size)
        self._consecutive_failures = 0
        self._error_count = 0
        self._last_result: Optional[ComponentHealth] = None

    def execute(self) -> ComponentHealth:
        """Execute the health check and return result."""
        start_time = time.time()
        status = HealthStatus.UNKNOWN
        message = ""
        details = {}

        try:
            result = self.check_fn()
            latency_ms = (time.time() - start_time) * 1000

            if result is True:
                status = HealthStatus.HEALTHY
                message = "Check passed"
                with self._lock:
                    self._consecutive_failures = 0
            elif isinstance(result, dict):
                # Allow detailed response
                status = HealthStatus(result.get("status", "healthy"))
                message = result.get("message", "")
                details = result.get("details", {})
                if status == HealthStatus.HEALTHY:
                    with self._lock:
                        self._consecutive_failures = 0
                else:
                    with self._lock:
                        self._consecutive_failures += 1
            else:
                status = HealthStatus.UNHEALTHY
                message = "Check returned false"
                with self._lock:
                    self._consecutive_failures += 1
                    self._error_count += 1

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            status = HealthStatus.UNHEALTHY
            message = f"Check failed: {str(e)}"
            logger.error(f"Health check '{self.name}' failed: {e}")
            with self._lock:
                self._consecutive_failures += 1
                self._error_count += 1

        # Determine final status based on consecutive failures
        with self._lock:
            if self._consecutive_failures >= self.config.failure_threshold:
                status = HealthStatus.UNHEALTHY
            elif self._consecutive_failures >= self.config.degraded_threshold:
                status = HealthStatus.DEGRADED

            health = ComponentHealth(
                name=self.name,
                status=status,
                message=message,
                last_check=start_time,
                latency_ms=latency_ms,
                details=details,
                dependencies=self.dependencies,
                error_count=self._error_count,
                consecutive_failures=self._consecutive_failures
            )

            self._history.append(health)
            self._last_result = health

        return health

    @property
    def last_result(self) -> Optional[ComponentHealth]:
        """Get the most recent health check result."""
        return self._last_result

    def get_history(self) -> List[ComponentHealth]:
        """Get health check history."""
        return list(self._history)


class HealthMonitor:
    """
    Central health monitoring system for TensorGuard.

    Manages health checks for all components and provides
    aggregated system health status.
    """

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._checks: Dict[str, HealthCheck] = {}
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._listeners: List[Callable[[Dict[str, ComponentHealth]], None]] = []

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], bool],
        config: Optional[HealthCheckConfig] = None,
        dependencies: Optional[List[str]] = None
    ) -> HealthCheck:
        """Register a new health check."""
        with self._lock:
            check = HealthCheck(
                name=name,
                check_fn=check_fn,
                config=config,
                dependencies=dependencies
            )
            self._checks[name] = check
            logger.info(f"Registered health check: {name}")
            return check

    def unregister_check(self, name: str):
        """Unregister a health check."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                logger.info(f"Unregistered health check: {name}")

    def add_listener(self, listener: Callable[[Dict[str, ComponentHealth]], None]):
        """Add a listener for health check results."""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable):
        """Remove a health check listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def check_all(self) -> Dict[str, ComponentHealth]:
        """Execute all health checks and return results."""
        results = {}
        with self._lock:
            checks = list(self._checks.values())

        for check in checks:
            results[check.name] = check.execute()

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(results)
            except Exception as e:
                logger.error(f"Health listener error: {e}")

        return results

    def check_single(self, name: str) -> Optional[ComponentHealth]:
        """Execute a single health check."""
        with self._lock:
            check = self._checks.get(name)
        if check:
            return check.execute()
        return None

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get aggregated system health status.

        Returns overall status based on all component healths,
        with hierarchical dependency consideration.
        """
        results = {}
        with self._lock:
            for name, check in self._checks.items():
                if check.last_result:
                    results[name] = check.last_result

        if not results:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks registered",
                "components": {},
                "timestamp": time.time()
            }

        # Aggregate status
        statuses = [h.status for h in results.values()]
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
            message = "One or more components unhealthy"
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
            message = "System operating in degraded mode"
        elif any(s == HealthStatus.UNKNOWN for s in statuses):
            overall_status = HealthStatus.DEGRADED
            message = "Some components status unknown"
        elif any(s == HealthStatus.INITIALIZING for s in statuses):
            overall_status = HealthStatus.INITIALIZING
            message = "System initializing"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All components healthy"

        return {
            "status": overall_status.value,
            "message": message,
            "components": {k: v.to_dict() for k, v in results.items()},
            "healthy_count": sum(1 for s in statuses if s == HealthStatus.HEALTHY),
            "degraded_count": sum(1 for s in statuses if s == HealthStatus.DEGRADED),
            "unhealthy_count": sum(1 for s in statuses if s == HealthStatus.UNHEALTHY),
            "total_count": len(statuses),
            "timestamp": time.time()
        }

    def get_dependency_tree(self) -> Dict[str, List[str]]:
        """Get the dependency tree of all components."""
        tree = {}
        with self._lock:
            for name, check in self._checks.items():
                tree[name] = check.dependencies
        return tree

    def check_dependencies_healthy(self, component_name: str) -> bool:
        """Check if all dependencies of a component are healthy."""
        with self._lock:
            check = self._checks.get(component_name)
            if not check:
                return True

            for dep_name in check.dependencies:
                dep_check = self._checks.get(dep_name)
                if dep_check and dep_check.last_result:
                    if dep_check.last_result.status == HealthStatus.UNHEALTHY:
                        return False
        return True

    def start_monitoring(self):
        """Start background health monitoring."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="health-monitor"
        )
        self._monitor_thread.start()
        logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop background health monitoring."""
        self._running = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self.check_all()
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")

            if self._stop_event.wait(timeout=self.check_interval):
                break

    def get_unhealthy_components(self) -> List[str]:
        """Get list of unhealthy component names."""
        unhealthy = []
        with self._lock:
            for name, check in self._checks.items():
                if check.last_result and check.last_result.status == HealthStatus.UNHEALTHY:
                    unhealthy.append(name)
        return unhealthy

    def get_degraded_components(self) -> List[str]:
        """Get list of degraded component names."""
        degraded = []
        with self._lock:
            for name, check in self._checks.items():
                if check.last_result and check.last_result.status == HealthStatus.DEGRADED:
                    degraded.append(name)
        return degraded


# Global health monitor instance
health_monitor = HealthMonitor()


def health_check(
    name: str,
    dependencies: Optional[List[str]] = None,
    interval: float = 30.0,
    failure_threshold: int = 3
):
    """
    Decorator for registering a function as a health check.

    Usage:
        @health_check("my_component")
        def check_my_component():
            return some_condition
    """
    def decorator(func: Callable) -> Callable:
        config = HealthCheckConfig(
            check_interval_seconds=interval,
            failure_threshold=failure_threshold
        )
        health_monitor.register_check(
            name=name,
            check_fn=func,
            config=config,
            dependencies=dependencies
        )
        return func
    return decorator
