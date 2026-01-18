"""
Policy Engine - Certificate Lifecycle Policy Evaluation

Evaluates certificates against policies to determine:
- Renewal eligibility
- EKU compliance (especially Jun 2026 Chrome rule)
- Algorithm constraints
- CA trust requirements
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import logging

from ..platform.models.identity_models import (
    IdentityCertificate,
    IdentityPolicy,
    IdentityEndpoint,
    CertificateType,
)

logger = logging.getLogger(__name__)


class PolicyViolation(str, Enum):
    """Types of policy violations."""
    EXPIRED = "expired"
    EXPIRING_SOON = "expiring_soon"
    EKU_CONFLICT = "eku_conflict"  # serverAuth + clientAuth on public cert
    WEAK_KEY = "weak_key"
    INVALID_KEY_TYPE = "invalid_key_type"
    INVALID_SIGNATURE_ALG = "invalid_signature_alg"
    UNTRUSTED_ISSUER = "untrusted_issuer"
    VALIDITY_TOO_LONG = "validity_too_long"
    MISSING_REQUIRED_EKU = "missing_required_eku"


@dataclass
class PolicyEvaluation:
    """Result of evaluating a certificate against a policy."""
    is_compliant: bool
    violations: List[PolicyViolation]
    needs_renewal: bool
    renewal_reason: Optional[str]
    days_to_expiry: int
    severity: str  # "critical", "warning", "info", "ok"
    details: Dict[str, Any]
    
    @property
    def is_critical(self) -> bool:
        return self.severity == "critical"
    
    @property
    def is_warning(self) -> bool:
        return self.severity == "warning"


class PolicyEngine:
    """
    Evaluates certificates against policies.
    
    Key features:
    - 47/100/200-day lifetime compliance
    - EKU separation enforcement (Jun 2026 Chrome rule)
    - Algorithm strength validation
    - Automatic renewal scheduling
    """
    
    # Chrome Jun 2026 cutoff
    EKU_SEPARATION_DEADLINE = datetime(2026, 6, 15)
    
    def __init__(self):
        self._custom_rules: List[callable] = []
    
    def evaluate(
        self,
        cert: IdentityCertificate,
        policy: IdentityPolicy,
        current_time: Optional[datetime] = None
    ) -> PolicyEvaluation:
        """
        Evaluate a certificate against a policy.
        
        Returns PolicyEvaluation with compliance status and violations.
        """
        current_time = current_time or datetime.utcnow()
        violations = []
        details = {}
        
        # Calculate days to expiry
        days_to_expiry = (cert.not_after - current_time).days
        details["days_to_expiry"] = days_to_expiry
        
        # 1. Check expiration
        if days_to_expiry <= 0:
            violations.append(PolicyViolation.EXPIRED)
        elif days_to_expiry <= policy.min_remaining_days:
            violations.append(PolicyViolation.EXPIRING_SOON)
        
        # 2. Check validity period
        cert_validity_days = (cert.not_after - cert.not_before).days
        if cert_validity_days > policy.max_validity_days:
            violations.append(PolicyViolation.VALIDITY_TOO_LONG)
            details["cert_validity_days"] = cert_validity_days
            details["max_allowed_days"] = policy.max_validity_days
        
        # 3. Check EKU (especially for Jun 2026 Chrome rule)
        eku_violations = self._check_eku(cert, policy, current_time)
        violations.extend(eku_violations)
        
        # 4. Check key strength
        key_violations = self._check_key_strength(cert, policy)
        violations.extend(key_violations)
        
        # 5. Check signature algorithm
        sig_violations = self._check_signature_algorithm(cert, policy)
        violations.extend(sig_violations)
        
        # 6. Check issuer trust
        issuer_violations = self._check_issuer(cert, policy)
        violations.extend(issuer_violations)
        
        # Determine severity
        if PolicyViolation.EXPIRED in violations:
            severity = "critical"
        elif PolicyViolation.EKU_CONFLICT in violations:
            severity = "critical"
        elif PolicyViolation.EXPIRING_SOON in violations:
            severity = "warning"
        elif len(violations) > 0:
            severity = "warning"
        else:
            severity = "ok"
        
        # Determine if renewal is needed
        needs_renewal = False
        renewal_reason = None
        
        if PolicyViolation.EXPIRED in violations:
            needs_renewal = True
            renewal_reason = "Certificate is expired"
        elif days_to_expiry <= policy.renewal_window_days:
            needs_renewal = True
            renewal_reason = f"Within renewal window ({policy.renewal_window_days} days)"
        elif PolicyViolation.EKU_CONFLICT in violations:
            needs_renewal = True
            renewal_reason = "EKU conflict requires certificate replacement"
        elif PolicyViolation.WEAK_KEY in violations:
            needs_renewal = True
            renewal_reason = "Key strength below policy minimum"
        
        return PolicyEvaluation(
            is_compliant=len(violations) == 0,
            violations=violations,
            needs_renewal=needs_renewal,
            renewal_reason=renewal_reason,
            days_to_expiry=days_to_expiry,
            severity=severity,
            details=details,
        )
    
    def _check_eku(
        self,
        cert: IdentityCertificate,
        policy: IdentityPolicy,
        current_time: datetime
    ) -> List[PolicyViolation]:
        """Check Extended Key Usage compliance."""
        violations = []
        
        # Check required EKUs
        if policy.allow_server_auth and not policy.allow_client_auth:
            if not cert.eku_server_auth:
                violations.append(PolicyViolation.MISSING_REQUIRED_EKU)
        
        if policy.allow_client_auth and not policy.allow_server_auth:
            if not cert.eku_client_auth:
                violations.append(PolicyViolation.MISSING_REQUIRED_EKU)
        
        # Check EKU separation (Chrome Jun 2026 rule)
        if policy.require_eku_separation and cert.is_public_trust:
            if cert.eku_server_auth and cert.eku_client_auth:
                # Violation is immediate if we're past deadline or close
                if current_time >= self.EKU_SEPARATION_DEADLINE:
                    violations.append(PolicyViolation.EKU_CONFLICT)
                elif (self.EKU_SEPARATION_DEADLINE - current_time).days <= 90:
                    # Warning: approaching deadline
                    violations.append(PolicyViolation.EKU_CONFLICT)
        
        return violations
    
    def _check_key_strength(
        self,
        cert: IdentityCertificate,
        policy: IdentityPolicy
    ) -> List[PolicyViolation]:
        """Check key type and strength."""
        violations = []
        
        # Parse allowed key types
        allowed_types = json.loads(policy.allowed_key_types_json)
        
        if cert.key_type.upper() not in [t.upper() for t in allowed_types]:
            violations.append(PolicyViolation.INVALID_KEY_TYPE)
            return violations
        
        # Check key size
        if cert.key_type.upper() == "RSA":
            if cert.key_size < policy.min_key_size_rsa:
                violations.append(PolicyViolation.WEAK_KEY)
        elif cert.key_type.upper() in ["ECDSA", "EC"]:
            if cert.key_size < policy.min_key_size_ec:
                violations.append(PolicyViolation.WEAK_KEY)
        
        return violations
    
    def _check_signature_algorithm(
        self,
        cert: IdentityCertificate,
        policy: IdentityPolicy
    ) -> List[PolicyViolation]:
        """Check signature algorithm is allowed."""
        violations = []
        
        allowed_algs = json.loads(policy.allowed_sig_algs_json)
        
        # Normalize and check
        cert_alg = cert.signature_algorithm.upper()
        allowed_normalized = [a.upper() for a in allowed_algs]
        
        # Check if any allowed alg is contained in cert alg
        if not any(alg in cert_alg for alg in allowed_normalized):
            violations.append(PolicyViolation.INVALID_SIGNATURE_ALG)
        
        return violations
    
    def _check_issuer(
        self,
        cert: IdentityCertificate,
        policy: IdentityPolicy
    ) -> List[PolicyViolation]:
        """Check issuer trust and restrictions."""
        violations = []
        
        # Check public trust requirement
        if policy.require_public_trust and not cert.is_public_trust:
            violations.append(PolicyViolation.UNTRUSTED_ISSUER)
        
        # Check allowed issuers if specified
        if policy.allowed_issuers_json:
            allowed_issuers = json.loads(policy.allowed_issuers_json)
            if allowed_issuers:
                if cert.issuer_dn not in allowed_issuers:
                    violations.append(PolicyViolation.UNTRUSTED_ISSUER)
        
        return violations
    
    def calculate_renewal_date(
        self,
        cert: IdentityCertificate,
        policy: IdentityPolicy
    ) -> datetime:
        """Calculate when a certificate should be renewed."""
        return cert.not_after - timedelta(days=policy.renewal_window_days)
    
    def get_expiry_bucket(self, cert: IdentityCertificate) -> str:
        """Categorize certificate by days to expiry."""
        days = cert.days_to_expiry
        
        if days <= 0:
            return "expired"
        elif days <= 7:
            return "critical"  # 0-7 days
        elif days <= 30:
            return "warning"   # 8-30 days
        elif days <= 60:
            return "attention" # 31-60 days
        elif days <= 90:
            return "upcoming"  # 61-90 days
        else:
            return "healthy"   # 90+ days
    
    def detect_eku_conflicts(
        self,
        certs: List[IdentityCertificate]
    ) -> List[Tuple[IdentityCertificate, str]]:
        """
        Detect certificates with EKU conflicts (Jun 2026 Chrome rule).
        
        Returns list of (certificate, recommendation) tuples.
        """
        conflicts = []
        
        for cert in certs:
            if cert.has_eku_conflict:
                recommendation = (
                    "Migrate to separate certificates: "
                    "public CA for serverAuth, private CA for clientAuth"
                )
                conflicts.append((cert, recommendation))
        
        return conflicts
