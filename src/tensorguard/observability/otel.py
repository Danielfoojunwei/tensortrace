"""
OpenTelemetry Setup for TensorGuard

Provides production-ready observability with:
- Distributed tracing via OpenTelemetry
- Metrics export to Prometheus
- Structured logging correlation
- Configurable exporters (OTLP, Console, Jaeger)

Configuration:
    TG_ENABLE_OTEL=true          Enable OpenTelemetry
    TG_OTEL_ENDPOINT=<url>       OTLP collector endpoint (default: localhost:4317)
    TG_OTEL_EXPORTER=otlp        Exporter type: otlp, console, jaeger
    TG_ENABLE_PROMETHEUS=true    Enable Prometheus metrics
    TG_PROMETHEUS_PORT=9090      Prometheus metrics port

Usage:
    from tensorguard.observability.otel import setup_observability, get_tracer

    # Initialize at application startup
    setup_observability("tensorguard-platform")

    # Get tracer for instrumentation
    tracer = get_tracer()
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("user_id", "123")
        # ... do work
"""

import os
import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Check for OpenTelemetry availability
OTEL_AVAILABLE = False
PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
        BatchSpanProcessor,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    OTEL_AVAILABLE = True
except ImportError:
    logger.warning("OpenTelemetry SDK not installed. Tracing disabled. Install: pip install opentelemetry-sdk")

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.info("prometheus_client not installed. Metrics disabled. Install: pip install prometheus-client")


# Global tracer instance
_tracer: Optional["trace.Tracer"] = None
_initialized = False


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

if PROMETHEUS_AVAILABLE:
    # Request metrics
    REQUEST_COUNT = Counter(
        'tensorguard_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status']
    )
    REQUEST_LATENCY = Histogram(
        'tensorguard_request_latency_seconds',
        'Request latency in seconds',
        ['method', 'endpoint'],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )

    # Training metrics
    TRAINING_ROUNDS = Counter(
        'tensorguard_training_rounds_total',
        'Total federated learning rounds',
        ['status']
    )
    DP_EPSILON_CONSUMED = Gauge(
        'tensorguard_dp_epsilon_consumed',
        'Cumulative differential privacy epsilon consumed',
        ['client_id']
    )

    # Crypto metrics
    ENCRYPTION_OPS = Counter(
        'tensorguard_encryption_operations_total',
        'Total encryption operations',
        ['operation', 'algorithm']
    )
    KEY_ROTATIONS = Counter(
        'tensorguard_key_rotations_total',
        'Total key rotation events',
        ['key_type']
    )

    # Health metrics
    ACTIVE_CONNECTIONS = Gauge(
        'tensorguard_active_connections',
        'Number of active client connections'
    )
    AGGREGATION_QUEUE_SIZE = Gauge(
        'tensorguard_aggregation_queue_size',
        'Number of pending client contributions'
    )


def setup_observability(
    service_name: str = "tensorguard",
    enable_tracing: bool = None,
    enable_metrics: bool = None,
) -> None:
    """
    Initialize observability stack (tracing + metrics).

    Args:
        service_name: Service name for trace attribution
        enable_tracing: Override TG_ENABLE_OTEL env var
        enable_metrics: Override TG_ENABLE_PROMETHEUS env var
    """
    global _tracer, _initialized

    if _initialized:
        logger.debug("Observability already initialized")
        return

    # Read from environment if not explicitly set
    if enable_tracing is None:
        enable_tracing = os.getenv("TG_ENABLE_OTEL", "false").lower() == "true"
    if enable_metrics is None:
        enable_metrics = os.getenv("TG_ENABLE_PROMETHEUS", "false").lower() == "true"

    # Setup tracing
    if enable_tracing and OTEL_AVAILABLE:
        _tracer = _setup_tracing(service_name)
        logger.info(f"OpenTelemetry tracing enabled for service: {service_name}")
    elif enable_tracing:
        logger.warning("Tracing requested but OpenTelemetry SDK not available")

    # Setup metrics
    if enable_metrics and PROMETHEUS_AVAILABLE:
        port = int(os.getenv("TG_PROMETHEUS_PORT", "9090"))
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    elif enable_metrics:
        logger.warning("Metrics requested but prometheus_client not available")

    _initialized = True


def _setup_tracing(service_name: str) -> "trace.Tracer":
    """Configure OpenTelemetry tracing."""
    # Create resource with service info
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_VERSION: os.getenv("TG_VERSION", "2.3.0"),
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("TG_ENVIRONMENT", "development"),
    })

    provider = TracerProvider(resource=resource)

    # Configure exporter based on environment
    exporter_type = os.getenv("TG_OTEL_EXPORTER", "console").lower()
    endpoint = os.getenv("TG_OTEL_ENDPOINT", "http://localhost:4317")

    if exporter_type == "otlp" and OTLP_AVAILABLE:
        exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = BatchSpanProcessor(exporter)
        logger.info(f"OTLP exporter configured: {endpoint}")
    else:
        exporter = ConsoleSpanExporter()
        processor = SimpleSpanProcessor(exporter)
        if exporter_type == "otlp":
            logger.warning("OTLP requested but exporter not available, using console")

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return trace.get_tracer(service_name)


def get_tracer(name: str = None) -> "trace.Tracer":
    """
    Get a tracer instance.

    Returns a no-op tracer if observability is not initialized or available.
    """
    global _tracer

    if _tracer is not None:
        return _tracer

    if OTEL_AVAILABLE:
        # Return default tracer if not explicitly configured
        return trace.get_tracer(name or "tensorguard")

    # Return a no-op tracer stub
    class NoOpSpan:
        def set_attribute(self, key, value): pass
        def add_event(self, name, attributes=None): pass
        def record_exception(self, exception): pass
        def set_status(self, status): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class NoOpTracer:
        def start_as_current_span(self, name, **kwargs):
            return NoOpSpan()
        def start_span(self, name, **kwargs):
            return NoOpSpan()

    return NoOpTracer()


@contextmanager
def trace_operation(name: str, attributes: dict = None):
    """
    Context manager for tracing an operation.

    Usage:
        with trace_operation("process_update", {"client_id": "123"}):
            # ... do work
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise


# Legacy compatibility
def setup_otel(service_name: str = "moai-inference"):
    """
    Legacy function for backward compatibility.
    Use setup_observability() for new code.
    """
    setup_observability(service_name, enable_tracing=True)
    return get_tracer(service_name)
