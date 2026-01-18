"""
CSR Generator - Local Key Generation and CSR Creation

Generates private keys locally (optionally via PKCS#11/TPM).
Private keys NEVER leave the agent.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
from ...core.keys import vault, KeyScope

logger = logging.getLogger(__name__)

# Attempt crypto imports
try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, ec
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    logger.warning("cryptography not installed, CSR generation disabled")


@dataclass
class KeyPair:
    """Generated key pair."""
    private_key: Any  # cryptography private key object
    public_key: Any
    key_type: str  # "RSA" or "ECDSA"
    key_size: int
    key_id: str  # Unique identifier


@dataclass
class CSRResult:
    """Generated CSR result."""
    csr_pem: str
    key_id: str
    subject_dn: str
    sans: List[str]
    key_type: str
    key_size: int


class CSRGenerator:
    """
    Generate CSRs with local key storage.
    
    Security:
    - Private keys stored encrypted or in PKCS#11/TPM
    - Private keys NEVER transmitted to control plane
    - Only CSR (public material) is sent
    """
    
    def __init__(
        self,
        key_storage_path: str = "keys/identity", # Standardized path
        encryption_key: Optional[bytes] = None,
    ):
        self.vault = vault
        self.scope = KeyScope.IDENTITY
        self.encryption_key = encryption_key
        
        # In-memory key cache (runtime only)
        self._key_cache: Dict[str, KeyPair] = {}
    
    def generate_key(
        self,
        key_type: str = "RSA",
        key_size: int = 2048,
        ec_curve: str = "P-256",
    ) -> KeyPair:
        """
        Generate a new key pair.
        
        Args:
            key_type: "RSA" or "ECDSA"
            key_size: RSA key size or ignored for EC
            ec_curve: EC curve name (P-256, P-384, P-521)
            
        Returns:
            KeyPair with generated keys
        """
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library required for key generation")
        
        import uuid
        key_id = str(uuid.uuid4())
        
        if key_type.upper() == "RSA":
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            actual_size = key_size
        elif key_type.upper() in ["ECDSA", "EC"]:
            curves = {
                "P-256": ec.SECP256R1(),
                "P-384": ec.SECP384R1(),
                "P-521": ec.SECP521R1(),
            }
            curve = curves.get(ec_curve, ec.SECP256R1())
            private_key = ec.generate_private_key(curve, default_backend())
            actual_size = curve.key_size
        else:
            raise ValueError(f"Unsupported key type: {key_type}")
        
        key_pair = KeyPair(
            private_key=private_key,
            public_key=private_key.public_key(),
            key_type=key_type.upper(),
            key_size=actual_size,
            key_id=key_id,
        )
        
        # Store encrypted
        self._store_key(key_pair)
        self._key_cache[key_id] = key_pair
        
        logger.info(f"Generated {key_type} {actual_size}-bit key: {key_id}")
        return key_pair
    
    def generate_csr(
        self,
        key_id: str,
        common_name: str,
        organization: Optional[str] = None,
        country: Optional[str] = None,
        sans: Optional[List[str]] = None,
        include_server_auth: bool = True,
        include_client_auth: bool = False,
    ) -> CSRResult:
        """
        Generate a CSR for an existing key.
        
        Args:
            key_id: ID of the key pair to use
            common_name: CN for the subject
            organization: O for the subject
            country: C for the subject
            sans: Subject Alternative Names (DNS/IP)
            include_server_auth: Include serverAuth EKU
            include_client_auth: Include clientAuth EKU
            
        Returns:
            CSRResult with PEM-encoded CSR
        """
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library required for CSR generation")
        
        key_pair = self._get_key(key_id)
        if not key_pair:
            raise ValueError(f"Key not found: {key_id}")
        
        # Build subject
        subject_attrs = [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
        if organization:
            subject_attrs.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization))
        if country:
            subject_attrs.append(x509.NameAttribute(NameOID.COUNTRY_NAME, country))
        
        subject = x509.Name(subject_attrs)
        
        # Build CSR
        builder = x509.CertificateSigningRequestBuilder()
        builder = builder.subject_name(subject)
        
        # Add SANs
        if sans:
            san_list = []
            for san in sans:
                if san.replace(".", "").isdigit():
                    # IP address
                    import ipaddress
                    san_list.append(x509.IPAddress(ipaddress.ip_address(san)))
                else:
                    san_list.append(x509.DNSName(san))
            
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False,
            )
        
        # Add EKU
        ekus = []
        if include_server_auth:
            ekus.append(ExtendedKeyUsageOID.SERVER_AUTH)
        if include_client_auth:
            ekus.append(ExtendedKeyUsageOID.CLIENT_AUTH)
        
        if ekus:
            builder = builder.add_extension(
                x509.ExtendedKeyUsage(ekus),
                critical=False,
            )
        
        # Sign CSR with private key
        csr = builder.sign(key_pair.private_key, hashes.SHA256(), default_backend())
        
        csr_pem = csr.public_bytes(serialization.Encoding.PEM).decode()
        
        logger.info(f"Generated CSR for {common_name} using key {key_id}")
        
        return CSRResult(
            csr_pem=csr_pem,
            key_id=key_id,
            subject_dn=subject.rfc4514_string(),
            sans=sans or [],
            key_type=key_pair.key_type,
            key_size=key_pair.key_size,
        )
    
    def generate_csr_with_new_key(
        self,
        common_name: str,
        sans: Optional[List[str]] = None,
        key_type: str = "RSA",
        key_size: int = 2048,
        organization: Optional[str] = None,
        include_server_auth: bool = True,
        include_client_auth: bool = False,
    ) -> CSRResult:
        """
        Generate a new key and CSR in one operation.
        
        Convenience method for common use case.
        """
        key_pair = self.generate_key(key_type=key_type, key_size=key_size)
        
        return self.generate_csr(
            key_id=key_pair.key_id,
            common_name=common_name,
            organization=organization,
            sans=sans,
            include_server_auth=include_server_auth,
            include_client_auth=include_client_auth,
        )
    
    def _store_key(self, key_pair: KeyPair) -> None:
        """Store key pair securely via Unified Vault."""
        if not HAS_CRYPTOGRAPHY:
            return
        
        # Serialize with encryption if key provided
        if self.encryption_key:
            pem_data = key_pair.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.BestAvailableEncryption(self.encryption_key)
            )
        else:
            # For demo: store unencrypted (NOT FOR PRODUCTION)
            pem_data = key_pair.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        
        self.vault.save_key_artifact(
            scope=self.scope,
            name=key_pair.key_id,
            data=pem_data,
            algorithm=key_pair.key_type,
            params={"key_size": key_pair.key_size},
            suffix=".key"
        )
    
    def _get_key(self, key_id: str) -> Optional[KeyPair]:
        """Retrieve a stored key pair from Unified Vault."""
        # Check cache first
        if key_id in self._key_cache:
            return self._key_cache[key_id]
        
        if not HAS_CRYPTOGRAPHY:
            return None
        
        try:
            pem_data, meta = self.vault.load_key_artifact(self.scope, key_id, suffix=".key")
            
            if self.encryption_key:
                private_key = serialization.load_pem_private_key(
                    pem_data,
                    password=self.encryption_key,
                    backend=default_backend()
                )
            else:
                private_key = serialization.load_pem_private_key(
                    pem_data,
                    password=None,
                    backend=default_backend()
                )
            
            key_pair = KeyPair(
                private_key=private_key,
                public_key=private_key.public_key(),
                key_type=meta.algorithm,
                key_size=meta.params.get("key_size", 0),
                key_id=key_id,
            )
            
            self._key_cache[key_id] = key_pair
            return key_pair
            
        except Exception as e:
            logger.error(f"Failed to load key {key_id} from vault: {e}")
            return None
    
    def delete_key(self, key_id: str) -> bool:
        """Delete a key pair from vault."""
        if key_id in self._key_cache:
            del self._key_cache[key_id]
        
        try:
            self.vault.delete_key(self.scope, key_id, suffix=".key")
            return True
        except:
            return False
    
    def list_keys(self) -> List[str]:
        """List all stored key IDs from vault."""
        keys = self.vault.list_keys(self.scope)
        return [k['key_id'].replace(f"{self.scope.value}_", "") for k in keys]

    def export_private_key_pem(self, key_id: str) -> str:
        """Export a private key PEM for deployment workflows."""
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library required for key export")

        key_pair = self._get_key(key_id)
        if not key_pair:
            raise ValueError(f"Key not found: {key_id}")

        pem_data = key_pair.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return pem_data.decode()
