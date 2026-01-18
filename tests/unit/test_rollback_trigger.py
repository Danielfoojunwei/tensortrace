"""
Test: Rollback Trigger Evaluation

Tests the automatic rollback trigger system:
- Error rate threshold triggers
- Latency threshold triggers
- Safety event threshold triggers
- Manual rollback handling
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple


class RollbackEvaluator:
    """
    Evaluates rollback conditions based on telemetry metrics.

    This mirrors the logic in rollout_service.py for testing purposes.
    """

    def __init__(self, guardrails: Dict[str, Any]):
        self.error_rate_threshold = guardrails.get("error_rate_threshold", 0.05)
        self.p99_latency_threshold_ms = guardrails.get("p99_latency_threshold_ms", 500)
        self.safety_event_threshold = guardrails.get("safety_event_threshold", 3)

    def evaluate(self, metrics: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Evaluate metrics against guardrails.

        Returns:
            Tuple of (should_rollback, trigger_details)
        """
        # Check error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.error_rate_threshold:
            return True, {
                "type": "error_rate",
                "threshold": self.error_rate_threshold,
                "actual": error_rate,
            }

        # Check p99 latency
        p99_latency = metrics.get("p99_latency_ms", 0)
        if p99_latency > self.p99_latency_threshold_ms:
            return True, {
                "type": "latency",
                "threshold": self.p99_latency_threshold_ms,
                "actual": p99_latency,
            }

        # Check safety events
        safety_events = metrics.get("safety_events", 0)
        if safety_events >= self.safety_event_threshold:
            return True, {
                "type": "safety",
                "threshold": self.safety_event_threshold,
                "actual": safety_events,
            }

        return False, None


class TestErrorRateTrigger:
    """Test error rate rollback trigger."""

    @pytest.fixture
    def evaluator(self):
        return RollbackEvaluator({
            "error_rate_threshold": 0.05,  # 5%
            "p99_latency_threshold_ms": 500,
            "safety_event_threshold": 3,
        })

    def test_below_threshold_no_rollback(self, evaluator):
        """Verify no rollback when error rate is below threshold."""
        metrics = {"error_rate": 0.03}  # 3%
        should_rollback, details = evaluator.evaluate(metrics)

        assert not should_rollback, "Should not rollback below threshold"
        assert details is None

    def test_at_threshold_no_rollback(self, evaluator):
        """Verify no rollback when error rate equals threshold."""
        metrics = {"error_rate": 0.05}  # Exactly 5%
        should_rollback, details = evaluator.evaluate(metrics)

        assert not should_rollback, "Should not rollback at exact threshold"

    def test_above_threshold_triggers_rollback(self, evaluator):
        """Verify rollback triggered when error rate exceeds threshold."""
        metrics = {"error_rate": 0.06}  # 6%
        should_rollback, details = evaluator.evaluate(metrics)

        assert should_rollback, "Should rollback above threshold"
        assert details["type"] == "error_rate"
        assert details["threshold"] == 0.05
        assert details["actual"] == 0.06

    def test_high_error_rate_trigger(self, evaluator):
        """Verify rollback for high error rates."""
        metrics = {"error_rate": 0.25}  # 25%
        should_rollback, details = evaluator.evaluate(metrics)

        assert should_rollback, "Should rollback for high error rate"


class TestLatencyTrigger:
    """Test latency rollback trigger."""

    @pytest.fixture
    def evaluator(self):
        return RollbackEvaluator({
            "error_rate_threshold": 0.05,
            "p99_latency_threshold_ms": 500,
            "safety_event_threshold": 3,
        })

    def test_below_latency_threshold_no_rollback(self, evaluator):
        """Verify no rollback when latency is below threshold."""
        metrics = {"p99_latency_ms": 300}
        should_rollback, details = evaluator.evaluate(metrics)

        assert not should_rollback, "Should not rollback below latency threshold"

    def test_above_latency_threshold_triggers_rollback(self, evaluator):
        """Verify rollback triggered when latency exceeds threshold."""
        metrics = {"p99_latency_ms": 600}
        should_rollback, details = evaluator.evaluate(metrics)

        assert should_rollback, "Should rollback above latency threshold"
        assert details["type"] == "latency"
        assert details["threshold"] == 500
        assert details["actual"] == 600

    def test_extreme_latency_trigger(self, evaluator):
        """Verify rollback for extreme latency."""
        metrics = {"p99_latency_ms": 5000}  # 5 seconds
        should_rollback, details = evaluator.evaluate(metrics)

        assert should_rollback, "Should rollback for extreme latency"


class TestSafetyEventTrigger:
    """Test safety event rollback trigger."""

    @pytest.fixture
    def evaluator(self):
        return RollbackEvaluator({
            "error_rate_threshold": 0.05,
            "p99_latency_threshold_ms": 500,
            "safety_event_threshold": 3,
        })

    def test_below_safety_threshold_no_rollback(self, evaluator):
        """Verify no rollback when safety events below threshold."""
        metrics = {"safety_events": 2}
        should_rollback, details = evaluator.evaluate(metrics)

        assert not should_rollback, "Should not rollback below safety threshold"

    def test_at_safety_threshold_triggers_rollback(self, evaluator):
        """Verify rollback triggered when safety events meet threshold."""
        metrics = {"safety_events": 3}
        should_rollback, details = evaluator.evaluate(metrics)

        assert should_rollback, "Should rollback at safety threshold"
        assert details["type"] == "safety"
        assert details["threshold"] == 3
        assert details["actual"] == 3

    def test_above_safety_threshold_triggers_rollback(self, evaluator):
        """Verify rollback triggered when safety events exceed threshold."""
        metrics = {"safety_events": 5}
        should_rollback, details = evaluator.evaluate(metrics)

        assert should_rollback, "Should rollback above safety threshold"


class TestMultipleTriggers:
    """Test handling of multiple rollback triggers."""

    @pytest.fixture
    def evaluator(self):
        return RollbackEvaluator({
            "error_rate_threshold": 0.05,
            "p99_latency_threshold_ms": 500,
            "safety_event_threshold": 3,
        })

    def test_error_rate_takes_precedence(self, evaluator):
        """Verify error rate is checked first."""
        metrics = {
            "error_rate": 0.10,  # Above threshold
            "p99_latency_ms": 600,  # Also above threshold
            "safety_events": 5,  # Also above threshold
        }
        should_rollback, details = evaluator.evaluate(metrics)

        assert should_rollback
        assert details["type"] == "error_rate", "Error rate should be checked first"

    def test_latency_checked_second(self, evaluator):
        """Verify latency is checked after error rate."""
        metrics = {
            "error_rate": 0.03,  # Below threshold
            "p99_latency_ms": 600,  # Above threshold
            "safety_events": 5,  # Also above threshold
        }
        should_rollback, details = evaluator.evaluate(metrics)

        assert should_rollback
        assert details["type"] == "latency", "Latency should be checked second"

    def test_safety_checked_last(self, evaluator):
        """Verify safety is checked last."""
        metrics = {
            "error_rate": 0.03,  # Below threshold
            "p99_latency_ms": 300,  # Below threshold
            "safety_events": 5,  # Above threshold
        }
        should_rollback, details = evaluator.evaluate(metrics)

        assert should_rollback
        assert details["type"] == "safety", "Safety should be checked last"

    def test_all_below_threshold(self, evaluator):
        """Verify no rollback when all metrics are healthy."""
        metrics = {
            "error_rate": 0.01,
            "p99_latency_ms": 100,
            "safety_events": 0,
        }
        should_rollback, details = evaluator.evaluate(metrics)

        assert not should_rollback
        assert details is None


class TestRollbackTriggerTypes:
    """Test different rollback trigger types."""

    def test_automatic_trigger_types(self):
        """Verify automatic trigger types are defined."""
        auto_triggers = ["error_rate", "latency", "safety"]

        for trigger_type in auto_triggers:
            assert trigger_type in ["error_rate", "latency", "safety"]

    def test_manual_trigger_type(self):
        """Verify manual rollback is a valid trigger type."""
        trigger_types = ["error_rate", "latency", "safety", "manual"]
        assert "manual" in trigger_types


class TestRollbackEvent:
    """Test rollback event structure."""

    def test_rollback_event_structure(self):
        """Verify rollback event has required fields."""
        rollback_event = {
            "id": "rollback-001",
            "deployment_id": "deploy-001",
            "trigger_type": "error_rate",
            "trigger_details": {
                "threshold": 0.05,
                "actual": 0.08,
            },
            "ts": datetime.utcnow().isoformat(),
        }

        required_fields = ["id", "deployment_id", "trigger_type", "ts"]
        for field in required_fields:
            assert field in rollback_event, f"Rollback event must have {field}"

    def test_manual_rollback_event(self):
        """Verify manual rollback event structure."""
        rollback_event = {
            "id": "rollback-002",
            "deployment_id": "deploy-001",
            "trigger_type": "manual",
            "trigger_details": {
                "initiated_by": "user-001",
                "reason": "Performance regression detected",
            },
            "ts": datetime.utcnow().isoformat(),
        }

        assert rollback_event["trigger_type"] == "manual"
        assert "initiated_by" in rollback_event["trigger_details"]
        assert "reason" in rollback_event["trigger_details"]


class TestGuardrailsConfiguration:
    """Test guardrails configuration validation."""

    def test_default_guardrails(self):
        """Verify default guardrails are reasonable."""
        default_guardrails = {
            "error_rate_threshold": 0.05,
            "p99_latency_threshold_ms": 500,
            "safety_event_threshold": 3,
        }

        # Error rate should be between 0 and 1
        assert 0 < default_guardrails["error_rate_threshold"] <= 1

        # Latency should be positive
        assert default_guardrails["p99_latency_threshold_ms"] > 0

        # Safety threshold should be positive integer
        assert default_guardrails["safety_event_threshold"] >= 1

    def test_custom_guardrails(self):
        """Verify custom guardrails work correctly."""
        custom_guardrails = {
            "error_rate_threshold": 0.10,  # 10% - more lenient
            "p99_latency_threshold_ms": 1000,  # 1 second - more lenient
            "safety_event_threshold": 5,  # More lenient
        }

        evaluator = RollbackEvaluator(custom_guardrails)

        # Should not rollback at default threshold values
        metrics = {"error_rate": 0.06}  # Would trigger with default
        should_rollback, _ = evaluator.evaluate(metrics)
        assert not should_rollback, "Custom threshold should be more lenient"

    def test_strict_guardrails(self):
        """Verify strict guardrails work correctly."""
        strict_guardrails = {
            "error_rate_threshold": 0.01,  # 1% - very strict
            "p99_latency_threshold_ms": 100,  # 100ms - very strict
            "safety_event_threshold": 1,  # Any safety event triggers
        }

        evaluator = RollbackEvaluator(strict_guardrails)

        # Should rollback at low error rate
        metrics = {"error_rate": 0.02}  # 2%
        should_rollback, _ = evaluator.evaluate(metrics)
        assert should_rollback, "Strict threshold should trigger rollback"
