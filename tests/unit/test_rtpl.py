"""
RTPL Unit Tests - Attack Pipeline and Defense Implementations

Tests for Robot Traffic Privacy Layer components.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile


class TestConvolutionDetector:
    """Test convolution-based command detection."""
    
    def test_default_kernels_exist(self):
        """Verify default kernels are defined."""
        from tensorguard.bench.rtpl.attack.convolution_detector import ConvolutionDetector
        
        detector = ConvolutionDetector()
        assert "cartesian" in detector.kernels
        assert "gripper_position" in detector.kernels
        assert "gripper_speed" in detector.kernels
    
    def test_convolve_detects_pattern(self):
        """Test that convolution detects known patterns."""
        from tensorguard.bench.rtpl.attack.convolution_detector import ConvolutionDetector
        
        detector = ConvolutionDetector(threshold=0.5)
        
        # Create signal with embedded pattern
        kernel = detector.kernels["cartesian"]
        signal = np.zeros(100)
        signal[40:40+len(kernel)] = kernel * 2  # Embed scaled pattern
        
        conv_output, detections = detector.convolve(signal, "cartesian")
        
        assert len(conv_output) == len(signal)
        # Should detect at least one pattern
        assert len(detections) >= 1
    
    def test_threshold_sweep(self):
        """Test threshold sweep functionality."""
        from tensorguard.bench.rtpl.attack.convolution_detector import ConvolutionDetector
        
        detector = ConvolutionDetector()
        signal = np.random.randn(200)
        
        results = detector.sweep_threshold(signal, "cartesian", [0.0, 0.5, 1.0])
        
        assert len(results) == 3
        # Lower threshold should give more detections
        assert results[0].num_detections >= results[2].num_detections


class TestCorrelationDetector:
    """Test correlation-based pattern detection."""
    
    def test_correlation_coefficient_range(self):
        """Verify correlation is in [-1, 1]."""
        from tensorguard.bench.rtpl.attack.correlation_detector import CorrelationDetector
        
        detector = CorrelationDetector()
        
        x = np.random.randn(50)
        y = np.random.randn(50)
        
        r = detector.compute_correlation(x, y)
        assert -1.0 <= r <= 1.0
    
    def test_perfect_correlation(self):
        """Test perfect correlation gives ~1.0."""
        from tensorguard.bench.rtpl.attack.correlation_detector import CorrelationDetector
        
        detector = CorrelationDetector()
        
        x = np.array([1, 2, 3, 4, 5])
        r = detector.compute_correlation(x, x)
        
        assert abs(r - 1.0) < 0.01


class TestFeatureExtractor:
    """Test feature extraction pipeline."""
    
    def test_feature_vector_length(self):
        """Verify correct number of features."""
        from tensorguard.bench.rtpl.attack.feature_extractor import ActionFeatures
        
        features = ActionFeatures()
        vector = features.to_vector()
        names = ActionFeatures.feature_names()
        
        assert len(vector) == len(names)
        assert len(vector) == 27  # Expected feature count


class TestPaddingDefense:
    """Test padding defense."""
    
    def test_padding_increases_size(self):
        """Verify padding increases packet size."""
        from tensorguard.agent.network.defense.padding import PaddingOnly, PaddingConfig
        
        defense = PaddingOnly(PaddingConfig(bucket_bytes=200))
        
        original = b"hello world"  # 11 bytes
        padded = defense.pad(original)
        
        # Should be padded to 200 bytes (with 4-byte header)
        assert len(padded) >= 200
    
    def test_padding_roundtrip(self):
        """Verify pad/strip roundtrip preserves data."""
        from tensorguard.agent.network.defense.padding import PaddingOnly
        
        defense = PaddingOnly()
        
        original = b"test data 123"
        padded = defense.pad(original)
        restored = defense.strip(padded)
        
        assert restored == original
    
    @pytest.mark.skip(reason="overhead_curve removed from production PaddingOnly")
    def test_overhead_curve(self):
        """Test overhead calculation for different bucket sizes."""
        pass


class TestFRONTDefense:
    """Test FRONT zero-delay defense."""
    
    def test_defend_adds_packets(self):
        """Verify FRONT adds dummy packets."""
        from tensorguard.agent.network.defense.front import FRONT
        
        defense = FRONT()
        
        original = [
            {"t": 0.0, "s": 100, "d": 1},
            {"t": 0.1, "s": 150, "d": -1},
        ]
        
        defended = defense.defend_trace(original)
        
        assert len(defended) > len(original)
    
    def test_rayleigh_timestamps(self):
        """Verify Rayleigh distribution bunches early."""
        from tensorguard.agent.network.defense.front import FRONT
        
        defense = FRONT()
        n, w, timestamps = defense.generate_schedule()
        
        # Timestamps should be sorted
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        
        # Most should be in early portion (Rayleigh property)
        median_idx = len(timestamps) // 2
        median_time = timestamps[median_idx]
        # Median should be around the scale parameter
        assert 0 < median_time < 10  # Reasonable range


class TestWTFPADDefense:
    """Test WTF-PAD zero-delay defense."""
    
    def test_histogram_sampling(self):
        """Verify histogram sampling works."""
        from tensorguard.agent.network.defense.wtf_pad import IATHistogram
        
        iats = np.array([0.01, 0.02, 0.03, 0.05, 0.1])
        hist = IATHistogram.from_samples(iats, n_bins=10)
        
        rng = np.random.default_rng(42)
        
        # Sample should be in reasonable range
        for _ in range(100):
            sample = hist.sample(rng)
            assert sample > 0
    
    @pytest.mark.skip(reason="simulate_defense removed from production WTFPAD")
    def test_simulate_defense(self):
        """Test offline simulation mode."""
        pass


class TestDeterminismGuard:
    """Test determinism safety controls."""
    
    def test_passes_within_bounds(self):
        """Verify passing when within bounds."""
        from tensorguard.agent.network.safety import DeterminismGuard
        
        guard = DeterminismGuard(max_added_latency_ms=10.0, max_jitter_ms=5.0)
        
        result = guard.check(measured_latency_ms=5.0, measured_jitter_ms=2.0)
        
        assert result.passed is True
    
    def test_fails_over_bounds(self):
        """Verify failing when exceeding bounds."""
        from tensorguard.agent.network.safety import DeterminismGuard
        
        guard = DeterminismGuard(max_added_latency_ms=10.0, max_jitter_ms=5.0)
        
        result = guard.check(measured_latency_ms=15.0, measured_jitter_ms=2.0)
        
        assert result.passed is False
        assert result.violation_type == "latency_exceeded"
    
    def test_profiles_exist(self):
        """Verify predefined profiles."""
        from tensorguard.agent.network.safety import PROFILES
        
        assert "surgical" in PROFILES
        assert "warehouse" in PROFILES
        assert "collaborative" in PROFILES


class TestSyntheticDataset:
    """Test synthetic dataset generation."""
    
    def test_generate_action_trace(self):
        """Verify trace generation."""
        from tensorguard.bench.rtpl.data.synthetic import SyntheticDatasetGenerator
        
        generator = SyntheticDatasetGenerator()
        flow = generator.generate_action_trace("pick_and_place")
        
        assert len(flow.packets) > 0
        assert flow.duration > 0
    
    def test_generate_dataset(self):
        """Verify full dataset generation."""
        from tensorguard.bench.rtpl.data.synthetic import SyntheticDatasetGenerator
        
        generator = SyntheticDatasetGenerator()
        flows, labels, class_names = generator.generate_dataset(samples_per_action=5)
        
        assert len(flows) == 5 * 4  # 5 samples * 4 classes
        assert len(labels) == len(flows)
        assert len(class_names) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
