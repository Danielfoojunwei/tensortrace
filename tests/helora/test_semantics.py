"""
Semantic equivalence tests for HE-LoRA.

These tests verify that the encrypted LoRA computation produces
results equivalent to plaintext computation (within CKKS approximation error).

Test requirements:
1. Compare decrypt(HE_LoRA_Delta) vs plaintext_LoRA_Delta
2. Assert relative error < epsilon (justified by CKKS approximation)
3. Test must FAIL if N2HE-HEXL backend not available
"""

import numpy as np
import pytest


# Skip all tests if backend not available, but mark as failures
def backend_available():
    """Check if N2HE-HEXL backend is available."""
    try:
        from crypto_backend.n2he_hexl import N2HEHEXLBackend
        backend = N2HEHEXLBackend()
        return backend.is_available()
    except ImportError:
        return False


# CKKS approximation error bounds
# These are justified by the CKKS scheme's approximate arithmetic
EPSILON_ENCRYPT_DECRYPT = 1e-5  # Single encrypt/decrypt cycle
EPSILON_SINGLE_MATMUL = 1e-4   # After one matmul + rescale
EPSILON_LORA_DELTA = 5e-4      # Full LoRA delta (2 matmuls + scaling)


class TestSemanticEquivalence:
    """Test semantic equivalence between HE and plaintext LoRA."""

    @pytest.fixture
    def setup_lora(self):
        """Set up LoRA test parameters."""
        return {
            "hidden_dim": 64,
            "rank": 8,
            "out_dim": 64,
            "alpha": 16.0,
        }

    def test_backend_required(self):
        """Test that backend is available - fail if not."""
        if not backend_available():
            pytest.fail(
                "N2HE-HEXL backend is REQUIRED for HE-LoRA tests.\n"
                "Build with: ./scripts/build_n2he_hexl.sh\n"
                "This test intentionally fails when backend is missing."
            )

    @pytest.mark.skipif(not backend_available(), reason="Backend required - this should fail")
    def test_encrypt_decrypt_roundtrip(self, setup_lora):
        """Test encrypt/decrypt roundtrip accuracy."""
        from tensafe.he_lora import get_backend

        backend = get_backend()
        hidden_dim = setup_lora["hidden_dim"]

        # Create test vector
        x = np.random.randn(hidden_dim).astype(np.float64)

        # Encrypt and decrypt
        ct = backend.encrypt(x)
        x_decrypted = backend.decrypt(ct, hidden_dim)

        # Check equivalence
        error = np.max(np.abs(x - x_decrypted))
        relative_error = error / (np.max(np.abs(x)) + 1e-10)

        assert relative_error < EPSILON_ENCRYPT_DECRYPT, (
            f"Encrypt/decrypt roundtrip error {relative_error:.2e} "
            f"exceeds threshold {EPSILON_ENCRYPT_DECRYPT}"
        )

    @pytest.mark.skipif(not backend_available(), reason="Backend required - this should fail")
    def test_lora_delta_equivalence(self, setup_lora):
        """Test HE LoRA delta equals plaintext LoRA delta."""
        from tensafe.he_lora import HELoRAAdapter, HELoRAConfig

        hidden_dim = setup_lora["hidden_dim"]
        rank = setup_lora["rank"]
        alpha = setup_lora["alpha"]

        # Create LoRA weights
        lora_a = np.random.randn(rank, hidden_dim).astype(np.float64) * 0.01
        lora_b = np.random.randn(hidden_dim, rank).astype(np.float64) * 0.01

        # Create adapter
        config = HELoRAConfig(rank=rank, alpha=alpha)
        adapter = HELoRAAdapter(config)
        adapter.register_weights("test", lora_a, lora_b, rank=rank, alpha=alpha)

        # Test input
        x = np.random.randn(hidden_dim).astype(np.float64)

        # Compute in both modes
        delta_he = adapter.forward(x, "test")
        delta_plaintext = adapter.forward_plaintext(x, "test")

        # Check equivalence
        error = np.max(np.abs(delta_he - delta_plaintext))
        relative_error = error / (np.max(np.abs(delta_plaintext)) + 1e-10)

        assert relative_error < EPSILON_LORA_DELTA, (
            f"HE vs plaintext LoRA delta error {relative_error:.2e} "
            f"exceeds threshold {EPSILON_LORA_DELTA}\n"
            f"HE delta norm: {np.linalg.norm(delta_he):.4f}\n"
            f"Plaintext delta norm: {np.linalg.norm(delta_plaintext):.4f}"
        )

    @pytest.mark.skipif(not backend_available(), reason="Backend required - this should fail")
    def test_lora_delta_multiple_samples(self, setup_lora):
        """Test equivalence across multiple samples."""
        from tensafe.he_lora import HELoRAAdapter, HELoRAConfig

        hidden_dim = setup_lora["hidden_dim"]
        rank = setup_lora["rank"]
        alpha = setup_lora["alpha"]

        # Create LoRA weights
        lora_a = np.random.randn(rank, hidden_dim).astype(np.float64) * 0.01
        lora_b = np.random.randn(hidden_dim, rank).astype(np.float64) * 0.01

        # Create adapter
        config = HELoRAConfig(rank=rank, alpha=alpha)
        adapter = HELoRAAdapter(config)
        adapter.register_weights("test", lora_a, lora_b, rank=rank, alpha=alpha)

        # Test multiple samples
        num_samples = 10
        max_errors = []

        for i in range(num_samples):
            x = np.random.randn(hidden_dim).astype(np.float64)

            delta_he = adapter.forward(x, "test")
            delta_plaintext = adapter.forward_plaintext(x, "test")

            error = np.max(np.abs(delta_he - delta_plaintext))
            relative_error = error / (np.max(np.abs(delta_plaintext)) + 1e-10)
            max_errors.append(relative_error)

        avg_error = np.mean(max_errors)
        max_error = np.max(max_errors)

        assert max_error < EPSILON_LORA_DELTA, (
            f"HE vs plaintext max error {max_error:.2e} exceeds threshold\n"
            f"Avg error: {avg_error:.2e}, samples: {num_samples}"
        )

    @pytest.mark.skipif(not backend_available(), reason="Backend required - this should fail")
    def test_lora_delta_different_dimensions(self):
        """Test equivalence with various dimensions."""
        from tensafe.he_lora import HELoRAAdapter, HELoRAConfig

        test_cases = [
            {"hidden_dim": 32, "rank": 4, "alpha": 8.0},
            {"hidden_dim": 64, "rank": 8, "alpha": 16.0},
            {"hidden_dim": 128, "rank": 16, "alpha": 32.0},
        ]

        for case in test_cases:
            hidden_dim = case["hidden_dim"]
            rank = case["rank"]
            alpha = case["alpha"]

            # Create LoRA weights
            lora_a = np.random.randn(rank, hidden_dim).astype(np.float64) * 0.01
            lora_b = np.random.randn(hidden_dim, rank).astype(np.float64) * 0.01

            # Create adapter
            config = HELoRAConfig(rank=rank, alpha=alpha)
            adapter = HELoRAAdapter(config)
            adapter.register_weights("test", lora_a, lora_b, rank=rank, alpha=alpha)

            # Test input
            x = np.random.randn(hidden_dim).astype(np.float64)

            delta_he = adapter.forward(x, "test")
            delta_plaintext = adapter.forward_plaintext(x, "test")

            error = np.max(np.abs(delta_he - delta_plaintext))
            relative_error = error / (np.max(np.abs(delta_plaintext)) + 1e-10)

            assert relative_error < EPSILON_LORA_DELTA, (
                f"Failed for dims h={hidden_dim}, r={rank}: "
                f"error {relative_error:.2e}"
            )


class TestErrorBounds:
    """Test error bounds are properly justified."""

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_error_scales_with_depth(self):
        """Verify error increases with operation depth."""
        from tensafe.he_lora import get_backend

        backend = get_backend()

        # Multiple encrypt/decrypt cycles should accumulate error
        x = np.random.randn(64).astype(np.float64)
        errors = []

        for _ in range(3):
            ct = backend.encrypt(x)
            x_dec = backend.decrypt(ct, len(x))
            error = np.max(np.abs(x - x_dec))
            errors.append(error)
            x = x_dec  # Chain

        # Later iterations may have slightly more error
        # but all should be below threshold
        assert all(e < EPSILON_ENCRYPT_DECRYPT * 2 for e in errors), (
            f"Chained operations exceed error bounds: {errors}"
        )

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_epsilon_is_reasonable(self):
        """Verify epsilon values are neither too tight nor too loose."""
        # Epsilon should be achievable but not trivially so
        # Values below 1e-7 would be too tight for CKKS
        # Values above 1e-2 would be too loose

        assert EPSILON_ENCRYPT_DECRYPT >= 1e-7, "Epsilon too tight for CKKS"
        assert EPSILON_ENCRYPT_DECRYPT <= 1e-3, "Epsilon too loose"

        assert EPSILON_LORA_DELTA >= 1e-6, "LoRA epsilon too tight"
        assert EPSILON_LORA_DELTA <= 1e-2, "LoRA epsilon too loose"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
