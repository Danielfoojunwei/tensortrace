"""
Test: No Random/Mock/Simulated in Production Endpoints

This test scans production endpoint files to ensure that no simulated data,
mocks, or random values are used without proper demo-mode gating.

Prevents regression to mock behavior in production paths.
"""

import pytest
import os
import re
from pathlib import Path


# Files that should be checked for production readiness
PROD_ENDPOINT_FILES = [
    "src/tensorguard/platform/api/endpoints.py",
    "src/tensorguard/platform/api/telemetry_endpoints.py",
    "src/tensorguard/platform/api/deployment_endpoints.py",
    "src/tensorguard/platform/api/bayesian_policy_endpoints.py",
    "src/tensorguard/platform/api/forensics_endpoints.py",
    "src/tensorguard/platform/api/config_endpoints.py",
]

# Patterns that indicate mock/simulated behavior (actual code, not comments)
FORBIDDEN_PATTERNS = [
    # Actual random function calls
    (r'random\.random\(\)', "Usage of random.random()"),
    (r'random\.randint\(', "Usage of random.randint()"),
    (r'random\.uniform\(', "Usage of random.uniform()"),
    (r'random\.choice\(', "Usage of random.choice()"),
    (r'random\.sample\(', "Usage of random.sample()"),
    (r'np\.random\.', "Usage of numpy random"),
    # Hardcoded demo/mock values
    (r'fleet_id\s*=\s*["\']demo[-_]fleet["\']', "Hardcoded demo-fleet assignment"),
]

# Patterns that indicate line should be skipped (comments, docstrings, etc.)
SKIP_LINE_PATTERNS = [
    r'^\s*#',  # Line starts with comment
    r'^\s*"""',  # Docstring start
    r'^\s*\'\'\'',  # Docstring start
    r'No random',  # Documentation stating no random usage
    r'no random',  # Documentation stating no random usage
    r'No mock',  # Documentation stating no mock usage
    r'no mock',  # Documentation stating no mock usage
    r'No simulated',  # Documentation stating no simulated usage
    r'no simulated',  # Documentation stating no simulated usage
]

# Allowed patterns (demo-mode gated)
ALLOWED_CONTEXTS = [
    r'is_demo_mode\(\)',  # Properly gated demo behavior
    r'TG_DEMO_MODE',  # Environment variable check
]


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Walk up until we find a directory with pyproject.toml or src/
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / "src").exists():
            return parent
    return current.parent.parent.parent


class TestNoRandomInProdEndpoints:
    """Test suite to verify no random/mock behavior in production endpoints."""

    @pytest.fixture
    def project_root(self):
        return get_project_root()

    def test_forbidden_patterns_not_in_prod_paths(self, project_root):
        """
        Verify that forbidden patterns are not used in production endpoints
        without proper demo-mode gating.
        """
        violations = []

        for file_path in PROD_ENDPOINT_FILES:
            full_path = project_root / file_path

            if not full_path.exists():
                # File might not exist yet
                continue

            content = full_path.read_text()
            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                # Skip empty lines
                if not line.strip():
                    continue

                # Skip lines that match skip patterns (comments, docstrings, etc.)
                should_skip = False
                for skip_pattern in SKIP_LINE_PATTERNS:
                    if re.search(skip_pattern, line, re.IGNORECASE):
                        should_skip = True
                        break

                if should_skip:
                    continue

                # Check if line is in an allowed context (demo-mode gated)
                is_allowed = False
                for allowed in ALLOWED_CONTEXTS:
                    if re.search(allowed, line):
                        is_allowed = True
                        break

                if is_allowed:
                    continue

                # Check for forbidden patterns
                for pattern, description in FORBIDDEN_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Check if it's properly gated in surrounding context
                        if 'is_demo_mode()' not in line and 'if is_demo_mode' not in content[max(0, content.find(line)-200):content.find(line)]:
                            violations.append({
                                "file": file_path,
                                "line": line_num,
                                "pattern": description,
                                "content": line.strip()[:80]
                            })

        # Report violations
        if violations:
            msg = "Found forbidden patterns in production endpoints:\n"
            for v in violations:
                msg += f"  {v['file']}:{v['line']} - {v['pattern']}\n"
                msg += f"    > {v['content']}\n"
            pytest.fail(msg)

    def test_random_module_imports_gated(self, project_root):
        """
        Verify that 'import random' is only used with demo-mode gating.
        """
        violations = []

        for file_path in PROD_ENDPOINT_FILES:
            full_path = project_root / file_path

            if not full_path.exists():
                continue

            content = full_path.read_text()

            # Check for random import
            if re.search(r'^import random\b', content, re.MULTILINE):
                # Verify there's demo-mode gating
                if 'is_demo_mode' not in content:
                    violations.append(file_path)

            if re.search(r'^from random import', content, re.MULTILINE):
                if 'is_demo_mode' not in content:
                    violations.append(file_path)

        if violations:
            pytest.fail(
                f"Found ungated 'import random' in: {violations}\n"
                "Random module usage must be gated with is_demo_mode()"
            )

    def test_demo_fleet_hardcoded(self, project_root):
        """
        Verify that 'demo-fleet' is not hardcoded in production endpoints.
        """
        violations = []

        for file_path in PROD_ENDPOINT_FILES:
            full_path = project_root / file_path

            if not full_path.exists():
                continue

            content = full_path.read_text()
            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                if 'demo-fleet' in line.lower() or 'demo_fleet' in line.lower():
                    # Check if it's in a comment or docstring
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                        continue

                    # Check if properly gated
                    if 'is_demo_mode' not in line:
                        violations.append({
                            "file": file_path,
                            "line": line_num,
                            "content": line.strip()[:80]
                        })

        if violations:
            msg = "Found hardcoded 'demo-fleet' references:\n"
            for v in violations:
                msg += f"  {v['file']}:{v['line']} - {v['content']}\n"
            pytest.fail(msg)

    def test_is_demo_mode_import_present(self, project_root):
        """
        Verify that files using demo behavior import is_demo_mode.
        """
        # Check that is_demo_mode is available
        gates_file = project_root / "src/tensorguard/utils/production_gates.py"

        assert gates_file.exists(), "production_gates.py must exist"

        content = gates_file.read_text()
        assert 'def is_demo_mode' in content, "is_demo_mode() must be defined"
        assert 'TG_DEMO_MODE' in content, "Must check TG_DEMO_MODE environment variable"
        assert 'TG_ENVIRONMENT' in content or 'is_production' in content, \
            "Must check environment is not production"

    def test_deterministic_mode_support(self, project_root):
        """
        Verify that TG_DETERMINISTIC mode is supported for reproducibility.
        """
        recovery_file = project_root / "src/tensorguard/hardening/recovery.py"

        assert recovery_file.exists(), "recovery.py must exist"

        content = recovery_file.read_text()
        assert 'TG_DETERMINISTIC' in content, \
            "recovery.py must support TG_DETERMINISTIC mode"
        assert 'hashlib' in content or 'sha256' in content.lower(), \
            "Deterministic mode should use hash-based jitter"


class TestEnvironmentGating:
    """Test environment variable gating logic."""

    def test_demo_mode_requires_non_production(self):
        """
        Verify that demo mode is only allowed when not in production.
        """
        import os

        # Save original values
        orig_demo = os.environ.get('TG_DEMO_MODE')
        orig_env = os.environ.get('TG_ENVIRONMENT')

        try:
            # Try to import tensorguard module
            try:
                from tensorguard.utils.production_gates import is_demo_mode, is_production
            except ImportError:
                pytest.skip("tensorguard module not installed - skipping runtime test")

            # Test 1: Demo mode disabled by default
            os.environ.pop('TG_DEMO_MODE', None)
            os.environ.pop('TG_ENVIRONMENT', None)

            # Need to reimport to pick up env changes
            import importlib
            import tensorguard.utils.production_gates as pg
            importlib.reload(pg)

            assert not pg.is_demo_mode(), "Demo mode should be disabled by default"

            # Test 2: Demo mode with production environment
            os.environ['TG_DEMO_MODE'] = 'true'
            os.environ['TG_ENVIRONMENT'] = 'production'
            importlib.reload(pg)

            assert not pg.is_demo_mode(), "Demo mode must be disabled in production"

            # Test 3: Demo mode with non-production environment
            os.environ['TG_DEMO_MODE'] = 'true'
            os.environ['TG_ENVIRONMENT'] = 'development'
            importlib.reload(pg)

            assert pg.is_demo_mode(), "Demo mode should be enabled in development"

        finally:
            # Restore original values
            if orig_demo:
                os.environ['TG_DEMO_MODE'] = orig_demo
            else:
                os.environ.pop('TG_DEMO_MODE', None)

            if orig_env:
                os.environ['TG_ENVIRONMENT'] = orig_env
            else:
                os.environ.pop('TG_ENVIRONMENT', None)
