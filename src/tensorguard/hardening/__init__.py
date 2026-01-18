"""
TensorGuard Hardening Module

Provides production-grade reliability, graceful degradation, and robustness
through circuit breakers, health monitoring, and recovery strategies.
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState
from .health_monitor import HealthMonitor, HealthStatus, ComponentHealth
from .recovery import RecoveryStrategy, RetryPolicy, FallbackHandler
from .telemetry import TelemetryCollector, MetricType, SystemMetrics
from .graceful_degradation import DegradationLevel, GracefulDegradationManager

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitState",
    "HealthMonitor",
    "HealthStatus",
    "ComponentHealth",
    "RecoveryStrategy",
    "RetryPolicy",
    "FallbackHandler",
    "TelemetryCollector",
    "MetricType",
    "SystemMetrics",
    "DegradationLevel",
    "GracefulDegradationManager",
]
