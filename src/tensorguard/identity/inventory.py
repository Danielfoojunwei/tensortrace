"""
Inventory Service - Certificate Discovery and Management

Provides certificate inventory management across endpoints.
Supports discovery via agent scans and manual registration.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlmodel import Session, select
import logging

from ..platform.models.identity_models import (
    IdentityEndpoint,
    IdentityCertificate,
    IdentityPolicy,
    IdentityAgent,
    EndpointType,
    Criticality,
    CertificateType,
)
from .policy_engine import PolicyEngine, PolicyEvaluation

logger = logging.getLogger(__name__)


class InventoryService:
    """
    Certificate inventory management.
    
    Features:
    - Endpoint CRUD
    - Certificate discovery and tracking
    - Expiry analysis and reporting
    - Risk assessment (blast radius)
    """
    
    def __init__(self, session: Session):
        self.session = session
        self.policy_engine = PolicyEngine()
    
    # === Endpoints ===
    
    def create_endpoint(
        self,
        tenant_id: str,
        fleet_id: str,
        name: str,
        hostname: str,
        port: int = 443,
        endpoint_type: EndpointType = EndpointType.CUSTOM,
        environment: str = "production",
        criticality: Criticality = Criticality.MEDIUM,
        k8s_namespace: Optional[str] = None,
        k8s_secret_name: Optional[str] = None,
        k8s_ingress_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> IdentityEndpoint:
        """Create a new endpoint."""
        import json
        
        endpoint = IdentityEndpoint(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            name=name,
            hostname=hostname,
            port=port,
            endpoint_type=endpoint_type,
            environment=environment,
            criticality=criticality,
            k8s_namespace=k8s_namespace,
            k8s_secret_name=k8s_secret_name,
            k8s_ingress_name=k8s_ingress_name,
            agent_id=agent_id,
            tags=json.dumps(tags) if tags else None,
        )
        
        self.session.add(endpoint)
        self.session.commit()
        self.session.refresh(endpoint)
        
        logger.info(f"Created endpoint: {endpoint.name} ({endpoint.hostname})")
        return endpoint
    
    def get_endpoint(self, endpoint_id: str) -> Optional[IdentityEndpoint]:
        """Get endpoint by ID."""
        return self.session.get(IdentityEndpoint, endpoint_id)
    
    def list_endpoints(
        self,
        tenant_id: str,
        fleet_id: Optional[str] = None,
        environment: Optional[str] = None,
        endpoint_type: Optional[EndpointType] = None,
        is_active: bool = True,
    ) -> List[IdentityEndpoint]:
        """List endpoints with filters."""
        statement = select(IdentityEndpoint).where(
            IdentityEndpoint.tenant_id == tenant_id,
            IdentityEndpoint.is_active == is_active
        )
        
        if fleet_id:
            statement = statement.where(IdentityEndpoint.fleet_id == fleet_id)
        if environment:
            statement = statement.where(IdentityEndpoint.environment == environment)
        if endpoint_type:
            statement = statement.where(IdentityEndpoint.endpoint_type == endpoint_type)
        
        return list(self.session.exec(statement).all())
    
    # === Certificates ===
    
    def register_certificate(
        self,
        endpoint_id: str,
        tenant_id: str,
        fingerprint_sha256: str,
        serial_number: str,
        subject_dn: str,
        issuer_dn: str,
        sans: List[str],
        not_before: datetime,
        not_after: datetime,
        key_type: str = "RSA",
        key_size: int = 2048,
        signature_algorithm: str = "SHA256WithRSA",
        eku_server_auth: bool = True,
        eku_client_auth: bool = False,
        is_public_trust: bool = True,
        is_self_signed: bool = False,
        certificate_type: CertificateType = CertificateType.PUBLIC_TLS,
        policy_id: Optional[str] = None,
    ) -> IdentityCertificate:
        """Register a discovered or issued certificate."""
        import json
        
        # Check for existing certificate
        existing = self.session.exec(
            select(IdentityCertificate).where(
                IdentityCertificate.fingerprint_sha256 == fingerprint_sha256
            )
        ).first()
        
        if existing:
            # Update last_seen_at
            existing.last_seen_at = datetime.utcnow()
            self.session.add(existing)
            self.session.commit()
            return existing
        
        cert = IdentityCertificate(
            endpoint_id=endpoint_id,
            tenant_id=tenant_id,
            fingerprint_sha256=fingerprint_sha256,
            serial_number=serial_number,
            subject_dn=subject_dn,
            issuer_dn=issuer_dn,
            sans_json=json.dumps(sans),
            not_before=not_before,
            not_after=not_after,
            key_type=key_type,
            key_size=key_size,
            signature_algorithm=signature_algorithm,
            eku_server_auth=eku_server_auth,
            eku_client_auth=eku_client_auth,
            is_public_trust=is_public_trust,
            is_self_signed=is_self_signed,
            certificate_type=certificate_type,
            policy_id=policy_id,
        )
        
        self.session.add(cert)
        self.session.commit()
        self.session.refresh(cert)
        
        logger.info(f"Registered certificate: {subject_dn} (expires: {not_after})")
        return cert
    
    def get_certificate(self, cert_id: str) -> Optional[IdentityCertificate]:
        """Get certificate by ID."""
        return self.session.get(IdentityCertificate, cert_id)
    
    def get_certificate_by_fingerprint(
        self,
        fingerprint: str
    ) -> Optional[IdentityCertificate]:
        """Get certificate by SHA-256 fingerprint."""
        statement = select(IdentityCertificate).where(
            IdentityCertificate.fingerprint_sha256 == fingerprint
        )
        return self.session.exec(statement).first()
    
    def list_certificates(
        self,
        tenant_id: str,
        fleet_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        expiry_within_days: Optional[int] = None,
        is_current: bool = True,
        certificate_type: Optional[CertificateType] = None,
    ) -> List[IdentityCertificate]:
        """List certificates with filters."""
        statement = select(IdentityCertificate).where(
            IdentityCertificate.tenant_id == tenant_id,
            IdentityCertificate.is_current == is_current
        )
        
        if endpoint_id:
            statement = statement.where(IdentityCertificate.endpoint_id == endpoint_id)
        
        if expiry_within_days is not None:
            expiry_threshold = datetime.utcnow() + timedelta(days=expiry_within_days)
            statement = statement.where(IdentityCertificate.not_after <= expiry_threshold)
        
        if certificate_type:
            statement = statement.where(IdentityCertificate.certificate_type == certificate_type)
        
        # Join with endpoint to filter by fleet
        if fleet_id:
            statement = statement.join(IdentityEndpoint).where(
                IdentityEndpoint.fleet_id == fleet_id
            )
        
        return list(self.session.exec(statement).all())
    
    # === Analysis ===
    
    def get_expiry_summary(
        self,
        tenant_id: str,
        fleet_id: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Get certificate counts by expiry bucket.
        
        Returns:
            Dict with keys: expired, critical (0-7d), warning (8-30d),
            attention (31-60d), upcoming (61-90d), healthy (90+d)
        """
        certs = self.list_certificates(tenant_id, fleet_id)
        
        buckets = {
            "expired": 0,
            "critical": 0,
            "warning": 0,
            "attention": 0,
            "upcoming": 0,
            "healthy": 0,
        }
        
        for cert in certs:
            bucket = self.policy_engine.get_expiry_bucket(cert)
            buckets[bucket] = buckets.get(bucket, 0) + 1
        
        return buckets
    
    def get_risk_assessment(
        self,
        tenant_id: str,
        cert_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess risk/blast radius if a certificate expires or fails.

        Returns:
            Dict with affected endpoints, criticality breakdown, and total risk score.
        """
        if cert_id:
            cert = self.get_certificate(cert_id)
            if not cert:
                return {"error": "Certificate not found"}
            certs = [cert]
        else:
            # Assess all expiring certificates
            certs = self.list_certificates(tenant_id, expiry_within_days=30)

        risk_data = {
            "certificates_at_risk": len(certs),
            "endpoints_affected": [],
            "criticality_breakdown": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            },
            "total_risk_score": 0,
        }

        # Criticality weights
        weights = {
            Criticality.CRITICAL: 10,
            Criticality.HIGH: 5,
            Criticality.MEDIUM: 2,
            Criticality.LOW: 1,
        }

        # Batch fetch all endpoints to avoid N+1 queries
        endpoint_ids = list(set(cert.endpoint_id for cert in certs))
        if endpoint_ids:
            endpoints_result = self.session.exec(
                select(IdentityEndpoint).where(IdentityEndpoint.id.in_(endpoint_ids))
            ).all()
            endpoints_map = {ep.id: ep for ep in endpoints_result}
        else:
            endpoints_map = {}

        for cert in certs:
            endpoint = endpoints_map.get(cert.endpoint_id)
            if endpoint:
                risk_data["endpoints_affected"].append({
                    "endpoint_id": endpoint.id,
                    "hostname": endpoint.hostname,
                    "environment": endpoint.environment,
                    "criticality": endpoint.criticality.value,
                    "days_to_expiry": cert.days_to_expiry,
                })

                risk_data["criticality_breakdown"][endpoint.criticality.value] += 1
                risk_data["total_risk_score"] += weights.get(endpoint.criticality, 1)

        return risk_data
    
    def detect_eku_violations(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Detect certificates with EKU conflicts for Jun 2026 Chrome rule.

        Returns list of certificates needing migration to private CA.
        """
        certs = self.list_certificates(tenant_id)

        # Filter certificates with EKU conflicts
        conflicting_certs = [cert for cert in certs if cert.has_eku_conflict]

        # Batch fetch all endpoints to avoid N+1 queries
        endpoint_ids = list(set(cert.endpoint_id for cert in conflicting_certs))
        if endpoint_ids:
            endpoints_result = self.session.exec(
                select(IdentityEndpoint).where(IdentityEndpoint.id.in_(endpoint_ids))
            ).all()
            endpoints_map = {ep.id: ep for ep in endpoints_result}
        else:
            endpoints_map = {}

        violations = []
        for cert in conflicting_certs:
            endpoint = endpoints_map.get(cert.endpoint_id)
            violations.append({
                "certificate_id": cert.id,
                "fingerprint": cert.fingerprint_sha256,
                "subject": cert.subject_dn,
                "endpoint": endpoint.name if endpoint else "Unknown",
                "days_to_expiry": cert.days_to_expiry,
                "recommendation": "Split into public (serverAuth) + private (clientAuth) certs",
            })

        return violations
    
    # === Agents ===
    
    def register_agent(
        self,
        tenant_id: str,
        fleet_id: str,
        name: str,
        hostname: str,
        supported_types: Optional[List[str]] = None,
        supported_challenges: Optional[List[str]] = None,
        has_pkcs11: bool = False,
        has_tpm: bool = False,
        public_key_pem: Optional[str] = None,
        version: Optional[str] = None,
    ) -> IdentityAgent:
        """Register a new identity agent."""
        import json
        
        agent = IdentityAgent(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            name=name,
            hostname=hostname,
            supported_types_json=json.dumps(supported_types or ["kubernetes", "nginx"]),
            supported_challenges_json=json.dumps(supported_challenges or ["http-01"]),
            has_pkcs11=has_pkcs11,
            has_tpm=has_tpm,
            public_key_pem=public_key_pem,
            version=version,
            last_heartbeat_at=datetime.utcnow(),
        )
        
        self.session.add(agent)
        self.session.commit()
        self.session.refresh(agent)
        
        logger.info(f"Registered agent: {agent.name} ({agent.hostname})")
        return agent
    
    def update_agent_heartbeat(self, agent_id: str) -> Optional[IdentityAgent]:
        """Update agent heartbeat timestamp."""
        agent = self.session.get(IdentityAgent, agent_id)
        if agent:
            agent.last_heartbeat_at = datetime.utcnow()
            self.session.add(agent)
            self.session.commit()
        return agent
    
    def list_agents(
        self,
        tenant_id: str,
        fleet_id: Optional[str] = None,
        is_active: bool = True,
    ) -> List[IdentityAgent]:
        """List agents with filters."""
        statement = select(IdentityAgent).where(
            IdentityAgent.tenant_id == tenant_id,
            IdentityAgent.is_active == is_active
        )
        
        if fleet_id:
            statement = statement.where(IdentityAgent.fleet_id == fleet_id)
        
        return list(self.session.exec(statement).all())

    def mark_cert_current(self, cert_id: str, endpoint_id: str):
        """Mark a certificate as the active one for an endpoint."""
        # 1. Unmark others
        others = self.session.exec(
            select(IdentityCertificate).where(
                IdentityCertificate.endpoint_id == endpoint_id,
                IdentityCertificate.id != cert_id,
                IdentityCertificate.is_current == True
            )
        ).all()
        for other in others:
            other.is_current = False
            self.session.add(other)
            
        # 2. Mark this one
        cert = self.get_certificate(cert_id)
        if cert:
            cert.is_current = True
            self.session.add(cert)
            
        self.session.commit()
        logger.info(f"Marked certificate {cert_id} as current for endpoint {endpoint_id}")
