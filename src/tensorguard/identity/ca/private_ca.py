"""
Private CA Integration - Internal Certificate Authority for mTLS

Supports integration with:
- Smallstep step-ca
- HashiCorp Vault PKI
- SPIFFE/SPIRE (for workload identity)
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrivateCAResult:
    """Result of private CA operations."""
    success: bool
    message: str
    certificate_pem: Optional[str] = None
    chain_pem: Optional[str] = None
    ca_cert_pem: Optional[str] = None


class PrivateCAClient:
    """
    Client for private CA integration.
    
    Used for:
    - mTLS client certificates (avoiding Chrome Jun 2026 EKU issue)
    - Internal service-to-service TLS
    - Workload identity (SPIFFE)
    """
    
    def __init__(
        self,
        ca_url: str,
        ca_type: str = "step-ca",  # "step-ca", "vault", "spire"
        auth_token: Optional[str] = None,
        provisioner: Optional[str] = None,
        verify_ssl: bool = True,
    ):
        self.ca_url = ca_url.rstrip("/")
        self.ca_type = ca_type
        self.auth_token = auth_token
        self.provisioner = provisioner
        self.verify_ssl = verify_ssl
    
    def issue_certificate(
        self,
        csr_pem: str,
        validity_hours: int = 720,  # 30 days
        san_dns: Optional[List[str]] = None,
        san_ip: Optional[List[str]] = None,
    ) -> PrivateCAResult:
        """
        Issue a certificate from the private CA.
        
        Args:
            csr_pem: PEM-encoded CSR
            validity_hours: Certificate validity in hours
            san_dns: Additional DNS SANs
            san_ip: Additional IP SANs
            
        Returns:
            PrivateCAResult with certificate
        """
        if self.ca_type == "step-ca":
            return self._issue_step_ca(csr_pem, validity_hours)
        elif self.ca_type == "vault":
            return self._issue_vault(csr_pem, validity_hours)
        elif self.ca_type == "spire":
            return self._issue_spire(csr_pem, validity_hours)
        else:
            return PrivateCAResult(
                success=False,
                message=f"Unsupported CA type: {self.ca_type}"
            )
    
    def _issue_step_ca(self, csr_pem: str, validity_hours: int) -> PrivateCAResult:
        """Issue certificate via step-ca."""
        try:
            # step-ca sign endpoint
            url = f"{self.ca_url}/1.0/sign"
            
            # Calculate not-after
            not_after = datetime.utcnow() + timedelta(hours=validity_hours)
            
            payload = {
                "csr": csr_pem,
                "ott": self.auth_token,  # One-time token for provisioner
                "notAfter": not_after.isoformat() + "Z",
            }
            
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                verify=self.verify_ssl,
                timeout=30,
            )
            
            if response.status_code == 200 or response.status_code == 201:
                data = response.json()
                return PrivateCAResult(
                    success=True,
                    message="Certificate issued by step-ca",
                    certificate_pem=data.get("crt"),
                    ca_cert_pem=data.get("ca"),
                )
            else:
                return PrivateCAResult(
                    success=False,
                    message=f"step-ca error: {response.text}"
                )
                
        except Exception as e:
            return PrivateCAResult(success=False, message=str(e))
    
    def _issue_vault(self, csr_pem: str, validity_hours: int) -> PrivateCAResult:
        """Issue certificate via HashiCorp Vault PKI."""
        try:
            # Vault PKI sign endpoint
            mount_path = "pki"  # Default mount path
            role = self.provisioner or "server"
            
            url = f"{self.ca_url}/v1/{mount_path}/sign/{role}"
            
            payload = {
                "csr": csr_pem,
                "ttl": f"{validity_hours}h",
            }
            
            headers = {}
            if self.auth_token:
                headers["X-Vault-Token"] = self.auth_token
            
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                verify=self.verify_ssl,
                timeout=30,
            )
            
            if response.status_code == 200:
                data = response.json()
                cert_data = data.get("data", {})
                
                # Vault returns certificate and ca_chain
                cert_pem = cert_data.get("certificate", "")
                ca_chain = cert_data.get("ca_chain", [])
                
                return PrivateCAResult(
                    success=True,
                    message="Certificate issued by Vault",
                    certificate_pem=cert_pem,
                    chain_pem="\n".join(ca_chain) if ca_chain else None,
                )
            else:
                return PrivateCAResult(
                    success=False,
                    message=f"Vault error: {response.text}"
                )
                
        except Exception as e:
            return PrivateCAResult(success=False, message=str(e))
    
    def _issue_spire(self, csr_pem: str, validity_hours: int) -> PrivateCAResult:
        """Issue certificate via SPIFFE/SPIRE."""
        # SPIRE typically uses workload API rather than CSR signing
        # This is a simplified stub
        return PrivateCAResult(
            success=False,
            message="SPIRE integration requires workload API, not CSR signing"
        )
    
    def get_ca_certificate(self) -> PrivateCAResult:
        """Get the CA's root/intermediate certificate for trust distribution."""
        if self.ca_type == "step-ca":
            return self._get_step_ca_root()
        elif self.ca_type == "vault":
            return self._get_vault_ca()
        else:
            return PrivateCAResult(
                success=False,
                message=f"Unsupported CA type: {self.ca_type}"
            )
    
    def _get_step_ca_root(self) -> PrivateCAResult:
        """Get step-ca root certificate."""
        try:
            url = f"{self.ca_url}/1.0/roots"
            
            response = requests.get(url, verify=self.verify_ssl, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                roots = data.get("crts", [])
                
                return PrivateCAResult(
                    success=True,
                    message="Retrieved CA root",
                    ca_cert_pem="\n".join(roots) if roots else None,
                )
            else:
                return PrivateCAResult(
                    success=False,
                    message=f"step-ca error: {response.text}"
                )
                
        except Exception as e:
            return PrivateCAResult(success=False, message=str(e))
    
    def _get_vault_ca(self) -> PrivateCAResult:
        """Get Vault PKI CA certificate."""
        try:
            mount_path = "pki"
            url = f"{self.ca_url}/v1/{mount_path}/ca/pem"
            
            response = requests.get(url, verify=self.verify_ssl, timeout=30)
            
            if response.status_code == 200:
                return PrivateCAResult(
                    success=True,
                    message="Retrieved CA certificate",
                    ca_cert_pem=response.text,
                )
            else:
                return PrivateCAResult(
                    success=False,
                    message=f"Vault error: {response.text}"
                )
                
        except Exception as e:
            return PrivateCAResult(success=False, message=str(e))


class EKUMigrationHelper:
    """
    Helper for migrating from public certs with serverAuth+clientAuth
    to separate certificates (Chrome Jun 2026 compliance).
    """
    
    def __init__(self, private_ca: PrivateCAClient):
        self.private_ca = private_ca
    
    def analyze_certificate(self, cert_pem: str) -> Dict[str, Any]:
        """
        Analyze a certificate for EKU migration needs.
        
        Returns analysis with migration recommendations.
        """
        try:
            from cryptography import x509
            from cryptography.hazmat.backends import default_backend
            
            cert = x509.load_pem_x509_certificate(cert_pem.encode(), default_backend())
            
            # Check EKU
            has_server_auth = False
            has_client_auth = False
            
            try:
                eku_ext = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
                for usage in eku_ext.value:
                    if usage == x509.oid.ExtendedKeyUsageOID.SERVER_AUTH:
                        has_server_auth = True
                    elif usage == x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH:
                        has_client_auth = True
            except x509.ExtensionNotFound:
                pass
            
            # Check issuer (public vs private)
            is_public = self._is_public_ca(cert.issuer.rfc4514_string())
            
            needs_migration = is_public and has_server_auth and has_client_auth
            
            return {
                "subject": cert.subject.rfc4514_string(),
                "issuer": cert.issuer.rfc4514_string(),
                "not_after": cert.not_valid_after.isoformat(),
                "has_server_auth": has_server_auth,
                "has_client_auth": has_client_auth,
                "is_public_trust": is_public,
                "needs_migration": needs_migration,
                "recommendation": (
                    "Split into two certificates:\n"
                    "1. Public CA cert with serverAuth only (for TLS)\n"
                    "2. Private CA cert with clientAuth only (for mTLS/VPN)"
                    if needs_migration else "No migration needed"
                ),
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _is_public_ca(self, issuer_dn: str) -> bool:
        """Check if issuer is a known public CA."""
        public_ca_keywords = [
            "Let's Encrypt",
            "DigiCert",
            "Sectigo",
            "GlobalSign",
            "GeoTrust",
            "Comodo",
            "ZeroSSL",
            "Buypass",
            "IdenTrust",
            "Amazon",
        ]
        
        return any(kw.lower() in issuer_dn.lower() for kw in public_ca_keywords)
    
    def generate_migration_plan(
        self,
        certificates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a migration plan for affected certificates.
        
        Args:
            certificates: List of certificate info dicts
            
        Returns:
            Migration plan with steps
        """
        affected = [c for c in certificates if c.get("needs_migration")]
        
        return {
            "total_certificates": len(certificates),
            "affected_certificates": len(affected),
            "deadline": "2026-06-15",
            "steps": [
                {
                    "step": 1,
                    "action": "Deploy private CA (step-ca or Vault)",
                    "details": "Set up internal CA infrastructure for client auth certs",
                },
                {
                    "step": 2,
                    "action": "Distribute CA trust anchor",
                    "details": "Push private CA root cert to all systems that verify client certs",
                },
                {
                    "step": 3,
                    "action": "Issue private client auth certs",
                    "details": "Generate new private CA certs with clientAuth EKU only",
                },
                {
                    "step": 4,
                    "action": "Update client configurations",
                    "details": "Configure clients to use new private certs for mTLS/VPN",
                },
                {
                    "step": 5,
                    "action": "Renew public certs without clientAuth",
                    "details": "Renew public certs with serverAuth only",
                },
                {
                    "step": 6,
                    "action": "Verify and cleanup",
                    "details": "Test all services and remove old dual-EKU certs",
                },
            ],
            "affected_services": [
                {
                    "subject": c.get("subject"),
                    "expires": c.get("not_after"),
                }
                for c in affected
            ],
        }
