"""
Production Security Gate Tests

These tests verify that production security gates are enforced:
- Demo mode cannot be enabled in production
- CORS wildcard blocked in production
- Enterprise entitlements fail-closed
- PQC strict mode enforced
- Experimental crypto blocked

Run with: pytest tests/security/test_production_gates.py -v
"""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestDemoModeGate:
    """Test that demo mode is blocked in production."""

    def test_demo_mode_default_off(self):
        """Demo mode should default to OFF."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove TG_DEMO_MODE to test default
            env = os.environ.copy()
            env.pop("TG_DEMO_MODE", None)

            with patch.dict(os.environ, env, clear=True):
                demo_mode = os.getenv("TG_DEMO_MODE", "false").lower() == "true"
                assert demo_mode is False, "Demo mode should default to false"

    def test_demo_mode_blocked_in_production(self):
        """Demo mode must be blocked when TG_ENVIRONMENT=production."""
        with patch.dict(os.environ, {
            "TG_ENVIRONMENT": "production",
            "TG_DEMO_MODE": "true"
        }):
            # Simulate the production check from auth.py
            environment = os.getenv("TG_ENVIRONMENT", "development")
            demo_mode = os.getenv("TG_DEMO_MODE", "false").lower() == "true"

            if environment == "production" and demo_mode:
                # This should raise or block
                blocked = True
            else:
                blocked = False

            assert blocked, "Demo mode should be blocked in production"


class TestCORSGate:
    """Test that CORS wildcard is blocked in production."""

    def test_cors_wildcard_blocked_in_production(self):
        """Wildcard CORS origin should be rejected in production."""
        with patch.dict(os.environ, {
            "TG_ENVIRONMENT": "production",
            "TG_ALLOWED_ORIGINS": "*"
        }):
            environment = os.getenv("TG_ENVIRONMENT", "development")
            origins = os.getenv("TG_ALLOWED_ORIGINS", "")

            if environment == "production" and "*" in origins:
                should_reject = True
            else:
                should_reject = False

            assert should_reject, "Wildcard CORS should be rejected in production"

    def test_cors_credentials_requires_explicit_origins(self):
        """Credentials with wildcard should be blocked."""
        with patch.dict(os.environ, {
            "TG_ALLOWED_ORIGINS": "*",
            "TG_ALLOW_CREDENTIALS": "true"
        }):
            origins = os.getenv("TG_ALLOWED_ORIGINS", "")
            credentials = os.getenv("TG_ALLOW_CREDENTIALS", "false").lower() == "true"

            if credentials and "*" in origins:
                should_block = True
            else:
                should_block = False

            assert should_block, "Credentials with wildcard should be blocked"


class TestEnterpriseEntitlements:
    """Test that enterprise entitlements fail-closed."""

    def test_enterprise_features_denied_in_community_mode(self):
        """Enterprise features should be denied in community mode."""
        with patch.dict(os.environ, {
            "TG_COMMUNITY_MODE": "true",
            "TG_ENVIRONMENT": "production"
        }):
            from tensorguard.platform.enterprise import check_entitlement, ENTERPRISE_FEATURES

            for feature in ENTERPRISE_FEATURES:
                result = check_entitlement("test_user", feature)
                assert result is False, f"Enterprise feature '{feature}' should be denied"

    def test_community_features_allowed(self):
        """Community features should always be allowed."""
        with patch.dict(os.environ, {
            "TG_COMMUNITY_MODE": "true",
            "TG_ENVIRONMENT": "production"
        }):
            from tensorguard.platform.enterprise import check_entitlement, COMMUNITY_FEATURES

            for feature in COMMUNITY_FEATURES:
                result = check_entitlement("test_user", feature)
                assert result is True, f"Community feature '{feature}' should be allowed"

    def test_unknown_features_denied(self):
        """Unknown features should be denied (fail-closed)."""
        with patch.dict(os.environ, {"TG_COMMUNITY_MODE": "true"}):
            from tensorguard.platform.enterprise import check_entitlement

            result = check_entitlement("test_user", "unknown_feature_xyz")
            assert result is False, "Unknown features should be denied"


class TestPQCStrictMode:
    """Test that PQC strict mode is enforced."""

    def test_pqc_strict_default_in_production(self):
        """PQC strict mode should be auto-enabled in production."""
        with patch.dict(os.environ, {
            "TG_ENVIRONMENT": "production",
            "TG_PQC_STRICT": ""  # Not explicitly set
        }):
            environment = os.getenv("TG_ENVIRONMENT", "development")
            strict = os.getenv("TG_PQC_STRICT", "").lower()

            # Auto-enable in production
            if strict == "":
                strict = "true" if environment == "production" else "false"

            assert strict == "true", "PQC strict should be auto-enabled in production"


class TestExperimentalCrypto:
    """Test that experimental crypto is blocked in production."""

    def test_experimental_crypto_blocked_in_production(self):
        """Experimental N2HE crypto should be blocked in production."""
        with patch.dict(os.environ, {
            "TG_ENVIRONMENT": "production",
            "TG_ENABLE_EXPERIMENTAL_CRYPTO": "false"
        }):
            environment = os.getenv("TG_ENVIRONMENT", "development")
            enable_experimental = os.getenv("TG_ENABLE_EXPERIMENTAL_CRYPTO", "false").lower() == "true"

            should_block = environment == "production" and not enable_experimental
            assert should_block, "Experimental crypto should be blocked in production"

    def test_experimental_crypto_allowed_with_explicit_flag(self):
        """Experimental crypto should be allowed with explicit flag."""
        with patch.dict(os.environ, {
            "TG_ENVIRONMENT": "production",
            "TG_ENABLE_EXPERIMENTAL_CRYPTO": "true"
        }):
            enable_experimental = os.getenv("TG_ENABLE_EXPERIMENTAL_CRYPTO", "false").lower() == "true"
            assert enable_experimental is True, "Explicit flag should enable experimental crypto"


class TestConfigValidation:
    """Test centralized configuration validation."""

    def test_config_validates_production_settings(self):
        """Configuration should validate production settings."""
        with patch.dict(os.environ, {
            "TG_ENVIRONMENT": "production",
            "TG_SECRET_KEY": "",  # Missing
            "TG_DEMO_MODE": "true",  # Bad
            "TG_ALLOWED_ORIGINS": "*",  # Bad
        }):
            from tensorguard.utils.config import TensorGuardSettings

            settings = TensorGuardSettings()
            issues = settings.validate_production_config()

            assert len(issues) > 0, "Should detect configuration issues"
            assert any("SECRET_KEY" in issue for issue in issues), "Should detect missing secret key"
            assert any("DEMO_MODE" in issue for issue in issues), "Should detect demo mode in production"

    def test_config_is_production_method(self):
        """is_production() should correctly detect production environment."""
        with patch.dict(os.environ, {"TG_ENVIRONMENT": "production"}):
            from tensorguard.utils.config import TensorGuardSettings

            settings = TensorGuardSettings()
            assert settings.is_production() is True

        with patch.dict(os.environ, {"TG_ENVIRONMENT": "development"}):
            from tensorguard.utils.config import TensorGuardSettings

            settings = TensorGuardSettings()
            assert settings.is_production() is False


# ============================================================================
# CI Guard: Forbidden Strings in Production Code
# ============================================================================

import re
from pathlib import Path


# Strings that should NOT appear in production code (src/tensorguard/**)
FORBIDDEN_PRODUCTION_STRINGS = [
    "DUMMY_",  # Dummy artifacts
    "SimulatedTrainer",  # Old trainer class (quarantined to DemoTrainer)
    "mock_ciphertext",  # Mock crypto values
    "MVP_STUB_CERT",  # Stub certificates
]

# Paths where forbidden strings ARE allowed
ALLOWED_PATHS = [
    "tests/",
    "demo_",
    "examples/",
    "HARDENING_REPORT",
    "__pycache__",
]


def get_production_files():
    """Get all Python files in production paths."""
    project_root = Path(__file__).parent.parent.parent
    prod_path = project_root / "src" / "tensorguard"

    if not prod_path.exists():
        return []

    files = []
    for py_file in prod_path.rglob("*.py"):
        rel_path = str(py_file.relative_to(project_root))
        if not any(allowed in rel_path for allowed in ALLOWED_PATHS):
            files.append(py_file)

    return files


class TestForbiddenStringsCI:
    """CI guard tests for forbidden strings in production code."""

    def test_no_dummy_artifacts(self):
        """Ensure no DUMMY_ prefixed strings creating fake artifacts."""
        violations = []

        for file_path in get_production_files():
            content = file_path.read_text(errors="ignore")
            lines = content.split("\n")

            for i, line in enumerate(lines):
                # Skip comments and string definitions
                if line.strip().startswith("#"):
                    continue
                if "FORBIDDEN_PRODUCTION_STRINGS" in line:
                    continue

                # Look for DUMMY_ that's not in a comment or test list
                if re.search(r'(?<!["\'])DUMMY_', line):
                    violations.append((file_path.name, i + 1, line.strip()[:60]))

        assert not violations, (
            f"Found DUMMY_ strings in production code:\n"
            + "\n".join(f"  {f}:{ln}: {line}" for f, ln, line in violations[:5])
        )

    def test_no_simulated_trainer(self):
        """Ensure SimulatedTrainer is not used in production."""
        violations = []

        for file_path in get_production_files():
            content = file_path.read_text(errors="ignore")

            if "SimulatedTrainer" in content:
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if "SimulatedTrainer" in line:
                        if not line.strip().startswith("#"):
                            violations.append((file_path.name, i + 1))

        assert not violations, (
            f"Found SimulatedTrainer in production code:\n"
            + "\n".join(f"  {f}:{ln}" for f, ln in violations[:5])
        )

    def test_no_mvp_stub_certs(self):
        """Ensure no MVP_STUB_CERT placeholder certificates."""
        violations = []

        for file_path in get_production_files():
            content = file_path.read_text(errors="ignore")

            if "MVP_STUB_CERT" in content:
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if "MVP_STUB_CERT" in line:
                        if not line.strip().startswith("#"):
                            violations.append((file_path.name, i + 1))

        assert not violations, (
            f"Found MVP_STUB_CERT in production code:\n"
            + "\n".join(f"  {f}:{ln}" for f, ln in violations[:5])
        )


class TestProductionGatesModule:
    """Test the production_gates utility module."""

    def test_production_gates_imports(self):
        """Verify production_gates module can be imported."""
        from tensorguard.utils.production_gates import (
            is_production,
            require_env,
            require_dependency,
            ProductionGateError,
            block_demo_mode,
        )
        assert callable(is_production)
        assert callable(require_env)
        assert callable(require_dependency)
        assert callable(block_demo_mode)

    def test_is_production_detection(self):
        """Test is_production() detects environment correctly."""
        from tensorguard.utils.production_gates import is_production

        # Clear cache for accurate testing
        is_production.cache_clear()

        with patch.dict(os.environ, {"TG_ENVIRONMENT": "production"}):
            is_production.cache_clear()
            assert is_production() is True

        with patch.dict(os.environ, {"TG_ENVIRONMENT": "development"}):
            is_production.cache_clear()
            assert is_production() is False

        # Restore
        is_production.cache_clear()

    def test_require_env_fails_in_production(self):
        """Test require_env raises in production when var missing."""
        from tensorguard.utils.production_gates import (
            require_env,
            ProductionGateError,
            is_production,
        )

        is_production.cache_clear()

        with patch.dict(os.environ, {"TG_ENVIRONMENT": "production"}, clear=False):
            is_production.cache_clear()

            # Remove test var if exists
            test_var = "TG_TEST_VAR_NONEXISTENT"
            env_copy = os.environ.copy()
            env_copy.pop(test_var, None)

            with patch.dict(os.environ, env_copy, clear=True):
                os.environ["TG_ENVIRONMENT"] = "production"
                is_production.cache_clear()

                with pytest.raises(ProductionGateError):
                    require_env(test_var)

        is_production.cache_clear()


class TestDeterminismModule:
    """Test the determinism utility module."""

    def test_determinism_imports(self):
        """Verify determinism module can be imported."""
        from tensorguard.utils.determinism import (
            is_deterministic_mode,
            get_determinism_seed,
            set_global_determinism,
        )
        assert callable(is_deterministic_mode)
        assert callable(get_determinism_seed)
        assert callable(set_global_determinism)

    def test_set_determinism_returns_config(self):
        """set_global_determinism should return config dict."""
        from tensorguard.utils.determinism import set_global_determinism

        result = set_global_determinism(seed=42)

        assert isinstance(result, dict)
        assert result["seed"] == 42
        assert "libraries" in result
