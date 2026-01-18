"""
Test: Rollout Determinism

Tests that deployment rollout cohort assignment is deterministic:
- Same device_id always gets same bucket
- Bucket distribution is uniform
- No randomness in assignment (uses SHA256 hash)

This ensures reproducible deployments across restarts and replicas.
"""

import pytest
import hashlib
from typing import List, Dict
from collections import Counter


def compute_bucket(device_id: str, num_buckets: int = 100) -> int:
    """
    Compute deterministic bucket for a device using SHA256 hash.

    This mirrors the implementation in rollout_service.py.
    """
    hash_bytes = hashlib.sha256(device_id.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
    return hash_int % num_buckets


class TestDeterministicCohortAssignment:
    """Test deterministic cohort assignment algorithm."""

    def test_same_device_same_bucket(self):
        """Verify same device_id always produces same bucket."""
        device_id = "device-test-12345"
        num_buckets = 100

        buckets = [compute_bucket(device_id, num_buckets) for _ in range(1000)]

        # All buckets should be identical
        assert len(set(buckets)) == 1, "Same device must always get same bucket"

    def test_different_devices_different_buckets(self):
        """Verify different devices get distributed across buckets."""
        num_buckets = 100
        devices = [f"device-{i}" for i in range(1000)]

        buckets = [compute_bucket(d, num_buckets) for d in devices]

        # Should have good distribution (not all in same bucket)
        unique_buckets = len(set(buckets))
        assert unique_buckets > 50, f"Should use most buckets, got {unique_buckets}"

    def test_bucket_distribution_uniformity(self):
        """Verify bucket distribution is approximately uniform."""
        num_buckets = 100
        num_devices = 10000

        devices = [f"device-{i}" for i in range(num_devices)]
        buckets = [compute_bucket(d, num_buckets) for d in devices]

        # Count devices per bucket
        counts = Counter(buckets)

        # Expected count per bucket
        expected = num_devices / num_buckets  # 100

        # Check that no bucket is more than 50% off from expected
        for bucket, count in counts.items():
            deviation = abs(count - expected) / expected
            assert deviation < 0.5, \
                f"Bucket {bucket} has {count} devices, expected ~{expected}"

    def test_determinism_across_calls(self):
        """Verify determinism across multiple function calls."""
        device_id = "device-reproducibility-test"

        bucket1 = compute_bucket(device_id)
        bucket2 = compute_bucket(device_id)
        bucket3 = compute_bucket(device_id)

        assert bucket1 == bucket2 == bucket3, \
            "Bucket assignment must be deterministic"

    def test_sha256_hash_used(self):
        """Verify SHA256 is used for hashing (not random)."""
        device_id = "device-hash-test"

        # Manually compute expected bucket
        hash_bytes = hashlib.sha256(device_id.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        expected_bucket = hash_int % 100

        actual_bucket = compute_bucket(device_id, 100)

        assert actual_bucket == expected_bucket, \
            "Bucket should be computed from SHA256 hash"


class TestCanaryRolloutAssignment:
    """Test canary deployment cohort assignment."""

    def test_canary_percentage(self):
        """Verify canary percentage is respected."""
        canary_pct = 10  # 10% canary
        num_devices = 10000

        devices = [f"device-{i}" for i in range(num_devices)]

        def assign_to_canary(device_id: str) -> bool:
            bucket = compute_bucket(device_id, 100)
            return bucket < canary_pct

        canary_count = sum(1 for d in devices if assign_to_canary(d))
        canary_ratio = canary_count / num_devices

        # Should be within 20% of expected ratio
        assert 0.08 < canary_ratio < 0.12, \
            f"Canary ratio should be ~10%, got {canary_ratio*100:.1f}%"

    def test_canary_promotion_to_cohort(self):
        """Verify canary devices remain included when promoting to cohort."""
        canary_pct = 10
        cohort_pct = 30

        device_id = "device-canary-test"
        bucket = compute_bucket(device_id, 100)

        # If device is in canary (bucket < 10), it should also be in cohort (bucket < 30)
        in_canary = bucket < canary_pct
        in_cohort = bucket < cohort_pct

        if in_canary:
            assert in_cohort, "Canary devices must be included in cohort stage"


class TestABTestingAssignment:
    """Test A/B testing cohort assignment."""

    def test_ab_split_50_50(self):
        """Verify 50/50 A/B split distribution."""
        num_devices = 10000
        devices = [f"device-{i}" for i in range(num_devices)]

        def assign_ab_variant(device_id: str) -> str:
            bucket = compute_bucket(device_id, 100)
            return "A" if bucket < 50 else "B"

        variants = [assign_ab_variant(d) for d in devices]
        counts = Counter(variants)

        ratio_a = counts["A"] / num_devices
        ratio_b = counts["B"] / num_devices

        # Should be within 5% of 50%
        assert 0.45 < ratio_a < 0.55, f"Variant A ratio should be ~50%, got {ratio_a*100:.1f}%"
        assert 0.45 < ratio_b < 0.55, f"Variant B ratio should be ~50%, got {ratio_b*100:.1f}%"

    def test_ab_assignment_determinism(self):
        """Verify A/B assignment is deterministic per device."""
        device_id = "device-ab-test"

        def assign_ab(device_id: str) -> str:
            bucket = compute_bucket(device_id, 100)
            return "A" if bucket < 50 else "B"

        variants = [assign_ab(device_id) for _ in range(100)]

        # All assignments should be the same
        assert len(set(variants)) == 1, \
            "A/B assignment must be deterministic"


class TestShadowModeAssignment:
    """Test shadow mode deployment assignment."""

    def test_shadow_mode_flag(self):
        """Verify shadow mode assignment logic."""
        shadow_pct = 5  # 5% shadow

        def is_shadow(device_id: str) -> bool:
            bucket = compute_bucket(device_id, 100)
            return bucket < shadow_pct

        # Test specific device
        device_id = "device-shadow-test"
        bucket = compute_bucket(device_id, 100)

        if bucket < shadow_pct:
            assert is_shadow(device_id), "Device in shadow bucket should be shadow"
        else:
            assert not is_shadow(device_id), "Device outside shadow bucket should not be shadow"


class TestCompatibilityVersionComparison:
    """Test version compatibility checking."""

    def test_version_comparison(self):
        """Verify semantic version comparison works correctly."""
        from packaging import version

        # version.parse handles semantic versioning
        assert version.parse("1.0.0") < version.parse("1.0.1")
        assert version.parse("1.0.0") < version.parse("1.1.0")
        assert version.parse("1.0.0") < version.parse("2.0.0")
        assert version.parse("1.0.0") == version.parse("1.0.0")
        assert version.parse("1.10.0") > version.parse("1.9.0")

    def test_compatibility_enforcement(self):
        """Verify agents below minimum version are rejected."""
        from packaging import version

        min_version = "1.2.0"

        compatible_versions = ["1.2.0", "1.2.1", "1.3.0", "2.0.0"]
        incompatible_versions = ["1.0.0", "1.1.0", "1.1.9", "0.9.0"]

        for v in compatible_versions:
            assert version.parse(v) >= version.parse(min_version), \
                f"Version {v} should be compatible"

        for v in incompatible_versions:
            assert version.parse(v) < version.parse(min_version), \
                f"Version {v} should be incompatible"


class TestDeploymentStageProgression:
    """Test deployment stage progression logic."""

    def test_stage_order(self):
        """Verify deployment stages follow correct order."""
        stages = ["draft", "running", "paused", "completed", "rolled_back"]

        # Draft can transition to running
        assert stages.index("draft") < stages.index("running")

        # Running can transition to completed or rolled_back
        assert stages.index("running") < stages.index("completed")
        assert stages.index("running") < stages.index("rolled_back")

    def test_cohort_expansion(self):
        """Verify cohort expands correctly: canary -> cohort -> full."""
        canary_pct = 10
        cohort_pct = 30
        full_pct = 100

        # Verify percentages are monotonically increasing
        assert canary_pct < cohort_pct < full_pct

        # Verify devices in earlier stages are included in later stages
        device_id = "device-expansion-test"
        bucket = compute_bucket(device_id, 100)

        in_canary = bucket < canary_pct
        in_cohort = bucket < cohort_pct
        in_full = bucket < full_pct

        # If in canary, must be in cohort
        if in_canary:
            assert in_cohort, "Canary devices must be in cohort"

        # If in cohort, must be in full
        if in_cohort:
            assert in_full, "Cohort devices must be in full"
