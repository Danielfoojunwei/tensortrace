"""
Prometheus Metrics for MOAI
"""

from prometheus_client import Counter, Histogram

# Request Counters
TOTAL_REQUESTS = Counter(
    "moai_inference_requests_total", 
    "Total FHE inference requests", 
    ["tenant_id", "status"]
)

# Latency Histograms
LATENCY_SECONDS = Histogram(
    "moai_inference_duration_seconds",
    "Time spent processing FHE inference",
    ["step"] # encrypt, compute, decrypt
)

# Key Usage
KEY_ROTATION_EVENTS = Counter(
    "moai_key_rotations_total",
    "Number of key rotation events",
    ["tenant_id"]
)
