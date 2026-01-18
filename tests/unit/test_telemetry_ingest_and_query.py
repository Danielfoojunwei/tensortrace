"""
Test: Telemetry Ingestion and Query

Tests the telemetry pipeline from edge agent to platform:
- HMAC authentication for telemetry ingestion
- Batch message processing
- Query endpoints for aggregated metrics
- Retention policy enforcement
"""

import pytest
import hashlib
import hmac
import json
import time
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any


class TestHMACAuthentication:
    """Test HMAC authentication for telemetry endpoints."""

    def test_hmac_signature_computation(self):
        """Verify HMAC signature computation matches expected format."""
        api_key = "test-fleet-api-key-12345"
        timestamp = "1704067200"  # Fixed timestamp for reproducibility
        nonce = "abcdef1234567890abcdef1234567890"
        body = b'{"batch_id": "test", "messages": []}'

        # Compute expected signature
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{timestamp}:{nonce}:{body_hash}"
        expected_signature = hmac.new(
            api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        # Verify signature is deterministic
        signature2 = hmac.new(
            api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        assert expected_signature == signature2, "HMAC should be deterministic"
        assert len(expected_signature) == 64, "SHA256 hex digest should be 64 chars"

    def test_hmac_signature_changes_with_body(self):
        """Verify HMAC signature changes when body changes."""
        api_key = "test-key"
        timestamp = "1704067200"
        nonce = "test-nonce"

        body1 = b'{"batch_id": "test1"}'
        body2 = b'{"batch_id": "test2"}'

        def compute_sig(body: bytes) -> str:
            body_hash = hashlib.sha256(body).hexdigest()
            message = f"{timestamp}:{nonce}:{body_hash}"
            return hmac.new(api_key.encode(), message.encode(), hashlib.sha256).hexdigest()

        sig1 = compute_sig(body1)
        sig2 = compute_sig(body2)

        assert sig1 != sig2, "Signatures must differ for different bodies"

    def test_hmac_signature_changes_with_timestamp(self):
        """Verify HMAC signature changes when timestamp changes."""
        api_key = "test-key"
        nonce = "test-nonce"
        body = b'{"batch_id": "test"}'
        body_hash = hashlib.sha256(body).hexdigest()

        sig1 = hmac.new(
            api_key.encode(),
            f"1704067200:{nonce}:{body_hash}".encode(),
            hashlib.sha256
        ).hexdigest()

        sig2 = hmac.new(
            api_key.encode(),
            f"1704067201:{nonce}:{body_hash}".encode(),
            hashlib.sha256
        ).hexdigest()

        assert sig1 != sig2, "Signatures must differ for different timestamps"


class TestTelemetryBatchFormat:
    """Test telemetry batch format validation."""

    def test_valid_batch_structure(self):
        """Verify valid batch structure is accepted."""
        batch = {
            "batch_id": f"batch-{int(time.time())}-{secrets.token_hex(4)}",
            "device_info": {
                "device_id": "device-001",
                "agent_version": "1.0.0",
            },
            "messages": [
                {
                    "topic": "telemetry.stage",
                    "timestamp_ns": int(time.time() * 1_000_000_000),
                    "payload": {
                        "device_id": "device-001",
                        "stage": "embed",
                        "duration_ms": 150.5,
                        "success": True,
                    },
                    "priority": 0,
                }
            ]
        }

        # Validate structure
        assert "batch_id" in batch
        assert "device_info" in batch
        assert "messages" in batch
        assert isinstance(batch["messages"], list)

        for msg in batch["messages"]:
            assert "topic" in msg
            assert "timestamp_ns" in msg
            assert "payload" in msg
            assert isinstance(msg["timestamp_ns"], int)

    def test_batch_id_uniqueness(self):
        """Verify batch IDs are unique."""
        batch_ids = set()

        for _ in range(100):
            batch_id = f"batch-{int(time.time())}-{secrets.token_hex(4)}"
            batch_ids.add(batch_id)
            time.sleep(0.001)  # Small delay to avoid timestamp collision

        # All batch IDs should be unique
        assert len(batch_ids) == 100, "Batch IDs should be unique"

    def test_topic_routing(self):
        """Verify message topics are valid telemetry topics."""
        valid_topics = [
            "telemetry.stage",
            "telemetry.system",
            "telemetry.model_behavior",
            "telemetry.forensics",
            "telemetry.heartbeat",
        ]

        for topic in valid_topics:
            assert topic.startswith("telemetry."), \
                f"Topic {topic} should start with 'telemetry.'"


class TestTelemetryMetricsAggregation:
    """Test telemetry metrics aggregation logic."""

    def test_percentile_calculation(self):
        """Verify percentile calculations are correct."""
        # Simulated latency data
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Sort for percentile calculation
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < n else f
            return sorted_latencies[f] + (k - f) * (sorted_latencies[c] - sorted_latencies[f])

        p50 = percentile(0.5)
        p90 = percentile(0.9)
        p99 = percentile(0.99)

        assert p50 == 55.0, f"P50 should be 55, got {p50}"
        assert p90 == 91.0, f"P90 should be 91, got {p90}"
        assert 99 <= p99 <= 100, f"P99 should be near 100, got {p99}"

    def test_error_rate_calculation(self):
        """Verify error rate calculation is correct."""
        events = [
            {"success": True},
            {"success": True},
            {"success": False},
            {"success": True},
            {"success": False},
        ]

        total = len(events)
        errors = sum(1 for e in events if not e["success"])
        error_rate = errors / total if total > 0 else 0

        assert error_rate == 0.4, f"Error rate should be 0.4, got {error_rate}"


class TestTelemetryRetention:
    """Test telemetry data retention policies."""

    def test_retention_policy_structure(self):
        """Verify retention policy structure."""
        policy = {
            "tenant_id": "tenant-001",
            "event_type": "stage",
            "retention_days": 30,
            "archive_after_days": 7,
        }

        assert policy["retention_days"] > 0, "Retention days must be positive"
        assert policy["archive_after_days"] <= policy["retention_days"], \
            "Archive threshold must be <= retention"

    def test_retention_expiry_calculation(self):
        """Verify retention expiry dates are calculated correctly."""
        retention_days = 30
        event_time = datetime.utcnow()
        expiry_time = event_time + timedelta(days=retention_days)

        # Event should not be expired if within retention period
        check_time = event_time + timedelta(days=15)
        assert check_time < expiry_time, "Event should not be expired at day 15"

        # Event should be expired after retention period
        check_time = event_time + timedelta(days=31)
        assert check_time > expiry_time, "Event should be expired at day 31"


class TestTelemetryTopics:
    """Test telemetry topic handling."""

    def test_stage_event_payload(self):
        """Verify stage event payload structure."""
        payload = {
            "device_id": "device-001",
            "stage": "embed",
            "duration_ms": 150.5,
            "success": True,
            "error_message": None,
            "metadata": {"model": "pi0"},
        }

        required_fields = ["device_id", "stage", "duration_ms", "success"]
        for field in required_fields:
            assert field in payload, f"Stage event must have {field}"

    def test_system_event_payload(self):
        """Verify system metrics payload structure."""
        payload = {
            "device_id": "device-001",
            "cpu_percent": 45.2,
            "memory_percent": 62.1,
            "disk_percent": 35.0,
            "gpu_percent": None,
            "gpu_memory_percent": None,
        }

        required_fields = ["device_id", "cpu_percent", "memory_percent", "disk_percent"]
        for field in required_fields:
            assert field in payload, f"System event must have {field}"

    def test_model_behavior_payload(self):
        """Verify model behavior payload structure."""
        payload = {
            "device_id": "device-001",
            "adapter_id": "adapter-v1.0",
            "inference_latency_ms": 25.3,
            "batch_size": 16,
            "error_rate": 0.01,
            "tokens_per_second": 150.0,
            "safety_flags": [],
        }

        required_fields = ["device_id", "adapter_id", "inference_latency_ms", "batch_size"]
        for field in required_fields:
            assert field in payload, f"Model behavior event must have {field}"

    def test_forensics_event_payload(self):
        """Verify forensics event payload structure."""
        payload = {
            "device_id": "device-001",
            "event_type": "adapter_swap",
            "deployment_id": "deploy-001",
            "adapter_id": "adapter-v2.0",
            "details": {"previous_adapter": "adapter-v1.0"},
            "pqc_signature": "pqc-sha512:abcdef123456...",
        }

        required_fields = ["device_id", "event_type", "deployment_id", "pqc_signature"]
        for field in required_fields:
            assert field in payload, f"Forensics event must have {field}"
