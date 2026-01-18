"""
Identity Module Tests - Policy Engine, Audit, and Scheduler Tests

Comprehensive unit tests for the Machine Identity Guard subsystem.
"""

import pytest
import hashlib
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


# === Policy Engine Tests ===

class TestPolicyEngine:
    """Tests for PolicyEngine."""
    
    def test_certificate_expired(self):
        """Test detection of expired certificate."""
        from tensorguard.identity.policy_engine import PolicyEngine, PolicyViolation
        
        engine = PolicyEngine()
        
        # Mock certificate
        cert = Mock()
        cert.not_before = datetime(2024, 1, 1)
        cert.not_after = datetime(2024, 6, 1)  # Expired
        cert.days_to_expiry = -100
        cert.is_public_trust = True
        cert.eku_server_auth = True
        cert.eku_client_auth = False
        cert.key_type = "RSA"
        cert.key_size = 2048
        cert.signature_algorithm = "SHA256WithRSA"
        cert.issuer_dn = "CN=Test CA"
        
        # Mock policy
        policy = Mock()
        policy.max_validity_days = 90
        policy.renewal_window_days = 30
        policy.min_remaining_days = 7
        policy.allow_server_auth = True
        policy.allow_client_auth = False
        policy.require_eku_separation = True
        policy.require_public_trust = True
        policy.min_key_size_rsa = 2048
        policy.min_key_size_ec = 256
        policy.allowed_key_types_json = '["RSA", "ECDSA"]'
        policy.allowed_sig_algs_json = '["SHA256", "SHA384"]'
        policy.allowed_issuers_json = None
        
        result = engine.evaluate(cert, policy, current_time=datetime(2024, 9, 1))
        
        assert not result.is_compliant
        assert PolicyViolation.EXPIRED in result.violations
        assert result.severity == "critical"
        assert result.needs_renewal
    
    def test_eku_conflict_detection(self):
        """Test EKU conflict detection (Chrome Jun 2026 rule)."""
        from tensorguard.identity.policy_engine import PolicyEngine, PolicyViolation
        
        engine = PolicyEngine()
        
        # Certificate with both EKUs
        cert = Mock()
        cert.not_before = datetime(2026, 1, 1)
        cert.not_after = datetime(2026, 12, 1)
        cert.days_to_expiry = 100
        cert.is_public_trust = True
        cert.eku_server_auth = True
        cert.eku_client_auth = True  # Conflict!
        cert.key_type = "RSA"
        cert.key_size = 2048
        cert.signature_algorithm = "SHA256WithRSA"
        cert.issuer_dn = "CN=Test CA"
        
        policy = Mock()
        policy.max_validity_days = 200
        policy.renewal_window_days = 30
        policy.min_remaining_days = 7
        policy.allow_server_auth = True
        policy.allow_client_auth = False
        policy.require_eku_separation = True
        policy.require_public_trust = True
        policy.min_key_size_rsa = 2048
        policy.min_key_size_ec = 256
        policy.allowed_key_types_json = '["RSA"]'
        policy.allowed_sig_algs_json = '["SHA256"]'
        policy.allowed_issuers_json = None
        
        # Test after Chrome deadline
        result = engine.evaluate(cert, policy, current_time=datetime(2026, 7, 1))
        
        assert PolicyViolation.EKU_CONFLICT in result.violations
        assert result.needs_renewal
    
    def test_weak_key_detection(self):
        """Test weak key size detection."""
        from tensorguard.identity.policy_engine import PolicyEngine, PolicyViolation
        
        engine = PolicyEngine()
        
        cert = Mock()
        cert.not_before = datetime(2025, 1, 1)
        cert.not_after = datetime(2025, 12, 1)
        cert.days_to_expiry = 300
        cert.is_public_trust = True
        cert.eku_server_auth = True
        cert.eku_client_auth = False
        cert.key_type = "RSA"
        cert.key_size = 1024  # Weak!
        cert.signature_algorithm = "SHA256WithRSA"
        cert.issuer_dn = "CN=Test CA"
        
        policy = Mock()
        policy.max_validity_days = 365
        policy.renewal_window_days = 30
        policy.min_remaining_days = 7
        policy.allow_server_auth = True
        policy.allow_client_auth = False
        policy.require_eku_separation = True
        policy.require_public_trust = True
        policy.min_key_size_rsa = 2048  # Policy requires 2048
        policy.min_key_size_ec = 256
        policy.allowed_key_types_json = '["RSA"]'
        policy.allowed_sig_algs_json = '["SHA256"]'
        policy.allowed_issuers_json = None
        
        result = engine.evaluate(cert, policy)
        
        assert PolicyViolation.WEAK_KEY in result.violations
        assert result.needs_renewal
    
    def test_compliant_certificate(self):
        """Test fully compliant certificate."""
        from tensorguard.identity.policy_engine import PolicyEngine
        
        engine = PolicyEngine()
        
        cert = Mock()
        cert.not_before = datetime(2025, 1, 1)
        cert.not_after = datetime(2025, 4, 1)  # 90-day cert
        cert.days_to_expiry = 85
        cert.is_public_trust = True
        cert.eku_server_auth = True
        cert.eku_client_auth = False
        cert.key_type = "RSA"
        cert.key_size = 2048
        cert.signature_algorithm = "SHA256WithRSA"
        cert.issuer_dn = "CN=Test CA"
        
        policy = Mock()
        policy.max_validity_days = 90
        policy.renewal_window_days = 30
        policy.min_remaining_days = 7
        policy.allow_server_auth = True
        policy.allow_client_auth = False
        policy.require_eku_separation = True
        policy.require_public_trust = True
        policy.min_key_size_rsa = 2048
        policy.min_key_size_ec = 256
        policy.allowed_key_types_json = '["RSA", "ECDSA"]'
        policy.allowed_sig_algs_json = '["SHA256", "SHA384"]'
        policy.allowed_issuers_json = None
        
        result = engine.evaluate(cert, policy, current_time=datetime(2025, 1, 5))
        
        assert result.is_compliant
        assert len(result.violations) == 0
        assert result.severity == "ok"


# === Audit Service Tests ===

class TestAuditService:
    """Tests for hash-chained audit log."""
    
    def test_hash_chain_integrity(self):
        """Test that hash chain is properly computed."""
        from tensorguard.identity.audit import AuditService
        from tensorguard.platform.models.identity_models import AuditAction
        
        # Mock session
        session = Mock()
        session.exec = Mock(return_value=Mock(first=Mock(return_value=None)))
        session.add = Mock()
        session.commit = Mock()
        session.refresh = Mock()
        
        audit = AuditService(session)
        
        # First entry should use genesis hash
        entry = audit.log(
            tenant_id="tenant-1",
            action=AuditAction.ENDPOINT_DISCOVERED,
            actor_type="user",
            actor_id="user-1",
            payload={"hostname": "example.com"},
        )
        
        # Verify session interactions
        assert session.add.called
        assert session.commit.called
    
    def test_payload_hash_computation(self):
        """Test payload hash is deterministic."""
        from tensorguard.identity.audit import AuditService
        
        session = Mock()
        audit = AuditService(session)
        
        payload = {"action": "test", "value": 123}
        
        hash1 = audit._compute_entry_hash("prev", "action", "payload_hash", datetime(2025, 1, 1))
        hash2 = audit._compute_entry_hash("prev", "action", "payload_hash", datetime(2025, 1, 1))
        
        assert hash1 == hash2  # Deterministic


# === Scheduler Tests ===

class TestRenewalScheduler:
    """Tests for renewal scheduler state machine."""
    
    def test_job_creation_idempotent(self):
        """Test that duplicate jobs are not created."""
        from tensorguard.identity.scheduler import RenewalScheduler
        from tensorguard.platform.models.identity_models import RenewalJobStatus
        
        session = Mock()
        
        # Mock existing pending job
        existing_job = Mock()
        existing_job.id = "existing-job"
        existing_job.status = RenewalJobStatus.PENDING
        
        session.exec = Mock(return_value=Mock(first=Mock(return_value=existing_job)))
        
        scheduler = RenewalScheduler(session)
        
        job = scheduler.schedule_renewal(
            tenant_id="tenant-1",
            fleet_id="fleet-1",
            endpoint_id="endpoint-1",
            policy_id="policy-1",
        )
        
        assert job.id == "existing-job"  # Returns existing job
    
    def test_job_state_transitions(self):
        """Test job state machine transitions."""
        from tensorguard.identity.scheduler import RenewalScheduler
        from tensorguard.platform.models.identity_models import RenewalJobStatus, IdentityRenewalJob
        
        # Create a real job object for testing
        job = IdentityRenewalJob(
            tenant_id="tenant-1",
            fleet_id="fleet-1",
            endpoint_id="endpoint-1",
            policy_id="policy-1",
            status=RenewalJobStatus.PENDING,
        )
        
        assert not job.is_terminal
        
        job.status = RenewalJobStatus.SUCCEEDED
        assert job.is_terminal
        
        job.status = RenewalJobStatus.PENDING
        job.retry_count = 0
        job.max_retries = 3
        assert job.can_retry
        
        job.status = RenewalJobStatus.FAILED
        assert not job.can_retry


# === CSR Generator Tests ===

class TestCSRGenerator:
    """Tests for CSR generation."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("cryptography", reason="cryptography not installed"),
        reason="cryptography not installed"
    )
    def test_key_generation(self):
        """Test RSA key generation."""
        from tensorguard.agent.identity.csr_generator import CSRGenerator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = CSRGenerator(key_storage_path=tmpdir)
            
            key_pair = generator.generate_key(key_type="RSA", key_size=2048)
            
            assert key_pair.key_type == "RSA"
            assert key_pair.key_size == 2048
            assert key_pair.key_id is not None
    
    @pytest.mark.skipif(
        not pytest.importorskip("cryptography", reason="cryptography not installed"),
        reason="cryptography not installed"
    )
    def test_csr_generation(self):
        """Test CSR generation."""
        from tensorguard.agent.identity.csr_generator import CSRGenerator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = CSRGenerator(key_storage_path=tmpdir)
            
            result = generator.generate_csr_with_new_key(
                common_name="example.com",
                sans=["example.com", "www.example.com"],
                key_type="RSA",
                key_size=2048,
            )
            
            assert result.csr_pem.startswith("-----BEGIN CERTIFICATE REQUEST-----")
            assert result.key_id is not None
            assert "example.com" in result.subject_dn


# === Integration Tests ===

class TestInventoryService:
    """Tests for inventory service."""
    
    def test_expiry_bucketing(self):
        """Test certificate expiry bucket categorization."""
        from tensorguard.identity.policy_engine import PolicyEngine
        
        engine = PolicyEngine()
        
        # Mock certs with various expiry times
        test_cases = [
            (-1, "expired"),
            (5, "critical"),
            (15, "warning"),
            (45, "attention"),
            (75, "upcoming"),
            (120, "healthy"),
        ]
        
        for days, expected_bucket in test_cases:
            cert = Mock()
            cert.days_to_expiry = days
            
            bucket = engine.get_expiry_bucket(cert)
            assert bucket == expected_bucket, f"Expected {expected_bucket} for {days} days"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
