"""
Telemetry Collection System

Comprehensive metrics collection, aggregation, and reporting for
system observability and performance monitoring.
"""

import time
import threading
import statistics
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Union
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"           # Monotonically increasing value
    GAUGE = "gauge"               # Point-in-time value
    HISTOGRAM = "histogram"       # Distribution of values
    TIMER = "timer"               # Duration measurements
    RATE = "rate"                 # Events per time period


@dataclass
class MetricConfig:
    """Configuration for metric collection."""
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    retention_seconds: float = 3600.0  # 1 hour default
    aggregation_interval: float = 60.0  # 1 minute buckets


@dataclass
class MetricValue:
    """A single metric measurement."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """Aggregated system metrics snapshot."""
    timestamp: float
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    active_connections: int = 0
    thread_count: int = 0
    process_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "memory_used_mb": self.memory_used_mb,
            "disk_usage_percent": self.disk_usage_percent,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "active_connections": self.active_connections,
            "thread_count": self.thread_count,
            "process_count": self.process_count,
        }


class Counter:
    """Thread-safe counter metric."""

    def __init__(self, config: MetricConfig):
        self.config = config
        self._value = 0.0
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=1000)

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment the counter."""
        with self._lock:
            self._value += value
            self._history.append(MetricValue(
                timestamp=time.time(),
                value=self._value,
                labels=labels or {}
            ))

    def get(self) -> float:
        """Get current counter value."""
        return self._value

    def reset(self):
        """Reset counter to zero."""
        with self._lock:
            self._value = 0.0


class Gauge:
    """Thread-safe gauge metric."""

    def __init__(self, config: MetricConfig):
        self.config = config
        self._value = 0.0
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=1000)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set the gauge value."""
        with self._lock:
            self._value = value
            self._history.append(MetricValue(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            ))

    def inc(self, value: float = 1.0):
        """Increment the gauge."""
        with self._lock:
            self._value += value

    def dec(self, value: float = 1.0):
        """Decrement the gauge."""
        with self._lock:
            self._value -= value

    def get(self) -> float:
        """Get current gauge value."""
        return self._value


class Histogram:
    """Thread-safe histogram metric with configurable buckets."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(self, config: MetricConfig, buckets: tuple = None):
        self.config = config
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._lock = threading.Lock()
        self._values: deque = deque(maxlen=10000)
        self._bucket_counts = {b: 0 for b in self.buckets}
        self._bucket_counts[float('inf')] = 0
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Record an observation."""
        with self._lock:
            self._values.append(MetricValue(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            ))
            self._sum += value
            self._count += 1

            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
                    break
            else:
                self._bucket_counts[float('inf')] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get histogram statistics."""
        with self._lock:
            values = [v.value for v in self._values]
            if not values:
                return {
                    "count": 0,
                    "sum": 0,
                    "mean": 0,
                    "min": 0,
                    "max": 0,
                    "p50": 0,
                    "p90": 0,
                    "p99": 0,
                    "buckets": dict(self._bucket_counts),
                }

            sorted_values = sorted(values)
            return {
                "count": self._count,
                "sum": self._sum,
                "mean": self._sum / self._count if self._count > 0 else 0,
                "min": min(values),
                "max": max(values),
                "p50": sorted_values[int(len(sorted_values) * 0.50)] if sorted_values else 0,
                "p90": sorted_values[int(len(sorted_values) * 0.90)] if sorted_values else 0,
                "p99": sorted_values[int(len(sorted_values) * 0.99)] if sorted_values else 0,
                "buckets": dict(self._bucket_counts),
            }


class Timer:
    """Thread-safe timer metric with context manager support."""

    def __init__(self, config: MetricConfig):
        self.config = config
        self._histogram = Histogram(config)

    def observe(self, duration_seconds: float, labels: Optional[Dict[str, str]] = None):
        """Record a duration observation."""
        self._histogram.observe(duration_seconds, labels)

    def time(self, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing a block of code."""
        return TimerContext(self, labels)

    def get_stats(self) -> Dict[str, Any]:
        """Get timer statistics."""
        return self._histogram.get_stats()


class TimerContext:
    """Context manager for Timer metric."""

    def __init__(self, timer: Timer, labels: Optional[Dict[str, str]] = None):
        self._timer = timer
        self._labels = labels
        self._start_time: Optional[float] = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self._start_time
        self._timer.observe(duration, self._labels)
        return False


class TelemetryCollector:
    """
    Central telemetry collection system for TensorGuard.

    Manages metrics collection, aggregation, and reporting.
    """

    _instance: Optional["TelemetryCollector"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._metrics: Dict[str, Union[Counter, Gauge, Histogram, Timer]] = {}
        self._metrics_lock = threading.RLock()
        self._running = False
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._system_metrics_history: deque = deque(maxlen=1000)
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._collection_interval = 15.0

        self._initialized = True

    def register_counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> Counter:
        """Register a new counter metric."""
        config = MetricConfig(
            name=name,
            metric_type=MetricType.COUNTER,
            description=description,
            labels=labels or {}
        )
        counter = Counter(config)
        with self._metrics_lock:
            self._metrics[name] = counter
        return counter

    def register_gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> Gauge:
        """Register a new gauge metric."""
        config = MetricConfig(
            name=name,
            metric_type=MetricType.GAUGE,
            description=description,
            labels=labels or {}
        )
        gauge = Gauge(config)
        with self._metrics_lock:
            self._metrics[name] = gauge
        return gauge

    def register_histogram(
        self,
        name: str,
        description: str = "",
        buckets: tuple = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Histogram:
        """Register a new histogram metric."""
        config = MetricConfig(
            name=name,
            metric_type=MetricType.HISTOGRAM,
            description=description,
            labels=labels or {}
        )
        histogram = Histogram(config, buckets)
        with self._metrics_lock:
            self._metrics[name] = histogram
        return histogram

    def register_timer(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> Timer:
        """Register a new timer metric."""
        config = MetricConfig(
            name=name,
            metric_type=MetricType.TIMER,
            description=description,
            labels=labels or {}
        )
        timer = Timer(config)
        with self._metrics_lock:
            self._metrics[name] = timer
        return timer

    def get_metric(self, name: str) -> Optional[Union[Counter, Gauge, Histogram, Timer]]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def add_listener(self, listener: Callable[[Dict[str, Any]], None]):
        """Add a listener for metric snapshots."""
        self._listeners.append(listener)

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        metrics = SystemMetrics(timestamp=time.time())

        try:
            import psutil

            # CPU
            metrics.cpu_usage_percent = psutil.cpu_percent(interval=0.1)

            # Memory
            mem = psutil.virtual_memory()
            metrics.memory_usage_percent = mem.percent
            metrics.memory_used_mb = mem.used / (1024 * 1024)

            # Disk
            disk = psutil.disk_usage('/')
            metrics.disk_usage_percent = disk.percent

            # Network
            net = psutil.net_io_counters()
            metrics.network_bytes_sent = net.bytes_sent
            metrics.network_bytes_recv = net.bytes_recv

            # Process info
            proc = psutil.Process()
            metrics.thread_count = proc.num_threads()

            # Connection count (simplified)
            try:
                metrics.active_connections = len(proc.connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                metrics.active_connections = 0

        except ImportError:
            logger.debug("psutil not available for system metrics")
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")

        self._system_metrics_history.append(metrics)
        return metrics

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get snapshot of all metrics."""
        snapshot = {
            "timestamp": time.time(),
            "metrics": {},
            "system": None
        }

        with self._metrics_lock:
            for name, metric in self._metrics.items():
                if isinstance(metric, Counter):
                    snapshot["metrics"][name] = {
                        "type": "counter",
                        "value": metric.get()
                    }
                elif isinstance(metric, Gauge):
                    snapshot["metrics"][name] = {
                        "type": "gauge",
                        "value": metric.get()
                    }
                elif isinstance(metric, (Histogram, Timer)):
                    snapshot["metrics"][name] = {
                        "type": metric.config.metric_type.value,
                        "stats": metric.get_stats()
                    }

        # Include latest system metrics
        if self._system_metrics_history:
            snapshot["system"] = self._system_metrics_history[-1].to_dict()

        return snapshot

    def get_system_metrics_history(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get system metrics history."""
        return [
            m.to_dict()
            for m in list(self._system_metrics_history)[-limit:]
        ]

    def start_collection(self, interval: float = 15.0):
        """Start background metrics collection."""
        if self._running:
            return

        self._collection_interval = interval
        self._running = True
        self._stop_event.clear()
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="telemetry-collector"
        )
        self._collection_thread.start()
        logger.info("Telemetry collection started")

    def stop_collection(self):
        """Stop background metrics collection."""
        self._running = False
        self._stop_event.set()
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        logger.info("Telemetry collection stopped")

    def _collection_loop(self):
        """Background collection loop."""
        while self._running:
            try:
                self.collect_system_metrics()

                # Notify listeners
                snapshot = self.get_all_metrics()
                for listener in self._listeners:
                    try:
                        listener(snapshot)
                    except Exception as e:
                        logger.error(f"Telemetry listener error: {e}")

            except Exception as e:
                logger.error(f"Telemetry collection error: {e}")

            if self._stop_event.wait(timeout=self._collection_interval):
                break

    def reset_all(self):
        """Reset all metrics."""
        with self._metrics_lock:
            for metric in self._metrics.values():
                if isinstance(metric, Counter):
                    metric.reset()


# Global telemetry collector instance
telemetry = TelemetryCollector()


# Convenience decorators
def timed(name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        timer = telemetry.register_timer(name, f"Execution time for {func.__name__}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timer.time(labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def counted(name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to count function calls."""
    def decorator(func: Callable) -> Callable:
        counter = telemetry.register_counter(name, f"Call count for {func.__name__}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            counter.inc(1.0, labels)
            return func(*args, **kwargs)
        return wrapper
    return decorator


import functools
