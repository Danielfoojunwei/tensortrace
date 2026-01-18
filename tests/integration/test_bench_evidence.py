import pytest
import os
import json
import shutil
from unittest.mock import MagicMock
from tensorguard.bench.cli import run_report
from tensorguard.bench.reporting import ReportGenerator

# Define path for artifacts
ARTIFACTS_DIR = "artifacts"

@pytest.fixture
def clean_artifacts():
    """Ensure artifacts directory is clean before and after test."""
    if os.path.exists(ARTIFACTS_DIR):
        shutil.rmtree(ARTIFACTS_DIR)
    os.makedirs(ARTIFACTS_DIR)
    yield
    # Cleanup after test if needed, or leave for inspection
    # shutil.rmtree(ARTIFACTS_DIR)

class TestBenchEvidence:
    """QA for Benchmark Reporting and Evidence Integrity."""

    @pytest.mark.perf
    def test_bench_report_generates_artifacts(self, clean_artifacts):
        """Test that the report command generates expected files."""
        # Create dummy data if required by run_report
        # Assuming run_report aggregates data from other artifact files
        # We might need to seed some dummy data files in artifacts/
        
        # Seed dummy metrics
        os.makedirs(f"{ARTIFACTS_DIR}/metrics", exist_ok=True)
        # Use .jsonl extension and proper naming
        with open(f"{ARTIFACTS_DIR}/metrics/micro_bench_1.jsonl", "w") as f:
            # Write valid JSONL
            f.write(json.dumps({"cpu_usage": 10.5, "memory_mb": 512, "fhe_ops_per_sec": 100, "metrics": {"latency_ms": 10}, "test_id": "test_1", "timestamp": 1234567890}) + "\n")
            
        args = MagicMock()
        run_report(args)

        assert os.path.exists(f"{ARTIFACTS_DIR}/report.html"), "report.html should exist"
        # report.json might not be implemented yet based on the request prompt implying "if implemented"
        # If it exists, verify it.
        if os.path.exists(f"{ARTIFACTS_DIR}/report.json"):
            with open(f"{ARTIFACTS_DIR}/report.json") as f:
                data = json.load(f)
                assert "run_id" in data
                assert "timestamp" in data
                assert "metrics" in data

    def test_bench_report_determinism(self, clean_artifacts):
        """Test that generating the report twice produces consistent output (excluding timestamps)."""
         # Seed dummy metrics
        os.makedirs(f"{ARTIFACTS_DIR}/metrics", exist_ok=True)
        with open(f"{ARTIFACTS_DIR}/metrics/micro_bench_1.jsonl", "w") as f:
            f.write(json.dumps({"cpu_usage": 10.5, "memory_mb": 512, "fhe_ops_per_sec": 100, "metrics": {"latency_ms": 10}, "test_id": "test_1", "timestamp": 1234567890}) + "\n")
            
        args = MagicMock()
        
        # Run 1
        run_report(args)
        if not os.path.exists(f"{ARTIFACTS_DIR}/report.json"):
            pytest.skip("report.json not implemented, cannot check deterministic output easily")
            
        with open(f"{ARTIFACTS_DIR}/report.json") as f:
            run1_data = json.load(f)
            
        # Run 2
        run_report(args)
        with open(f"{ARTIFACTS_DIR}/report.json") as f:
            run2_data = json.load(f)
            
        # Remove volatile fields
        run1_data.pop("timestamp", None)
        run1_data.pop("run_id", None)
        run1_data.pop("artifacts_hashes", None)
    
        run2_data.pop("timestamp", None)
        run2_data.pop("run_id", None)
        run2_data.pop("artifacts_hashes", None)
        
        # Verify details determinism
        # Sort metrics list if present to handle file system ordering differences
        if "metrics" in run1_data and isinstance(run1_data["metrics"], list):
            run1_data["metrics"].sort(key=lambda x: str(x))
        if "metrics" in run2_data and isinstance(run2_data["metrics"], list):
            run2_data["metrics"].sort(key=lambda x: str(x))
        
        assert run1_data == run2_data, "Report generation must be deterministic given same inputs"

    def test_artifact_hashing_and_canonicalization(self):
        """Ensure SHA-256 generation matches docs and is stable."""
        from tensorguard.tgsp import crypto
        import hashlib
        
        # Test Case 1: Known String
        data = b"test_string"
        
        # We verify that our crypto wrapper matches standard hashlib
        # This handles platform surprises if any
        expected = hashlib.sha256(data).hexdigest()
        actual = crypto.get_sha256(data)
        
        assert actual == expected, (
            f"Hash wrapper mismatch!\n"
            f"Input: {data}\n"
            f"Expected (hashlib): {expected}\n"
            f"Got (wrapper): {actual}"
        )
        
        # Test Case 2: Canonical Serialization (Dictionary)
        # Assuming we have a canonicalizer helper. If not, we test manifest serialization.
        from tensorguard.tgsp.manifest import PackageManifest
        m = PackageManifest()
        # Ensure it produces stable bytes
        b1 = m.to_canonical_cbor()
        b2 = m.to_canonical_cbor()
        assert b1 == b2

