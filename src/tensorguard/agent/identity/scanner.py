"""
Agent Scanner - Certificate Discovery on Endpoints

Scans local systems for TLS certificates:
- Kubernetes secrets
- Nginx configurations
- Envoy configurations
- File system (PEM files)
"""

import os
import ssl
import socket
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import base64
import json
import logging

logger = logging.getLogger(__name__)

# Attempt crypto imports
try:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    logger.warning("cryptography not installed, some features disabled")


@dataclass
class DiscoveredCertificate:
    """A certificate discovered during scanning."""
    source: str  # "kubernetes", "nginx", "file", "network"
    source_path: str  # Secret name, file path, host:port
    
    # Certificate data
    fingerprint_sha256: str
    serial_number: str
    subject_dn: str
    issuer_dn: str
    sans: List[str]
    not_before: datetime
    not_after: datetime
    
    # Key info
    key_type: str
    key_size: int
    signature_algorithm: str
    
    # EKU
    eku_server_auth: bool
    eku_client_auth: bool
    
    # Trust
    is_self_signed: bool
    
    # Raw data (for registration)
    pem_data: Optional[str] = None


class CertificateScanner:
    """
    Scan various sources for TLS certificates.
    
    Supports:
    - Kubernetes TLS secrets
    - Nginx server blocks
    - Envoy listener configurations
    - File system PEM files
    - Network TLS handshakes
    """
    
    def __init__(self, kubeconfig: Optional[str] = None):
        self.kubeconfig = kubeconfig
        self._k8s_client = None
    
    # === Kubernetes Scanning ===
    
    def scan_kubernetes(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None,
    ) -> List[DiscoveredCertificate]:
        """
        Scan Kubernetes TLS secrets.
        
        Looks for secrets of type kubernetes.io/tls.
        """
        certs = []
        
        try:
            # Try using kubectl
            cmd = ["kubectl", "get", "secrets", "-o", "json"]
            if namespace:
                cmd.extend(["-n", namespace])
            else:
                cmd.append("--all-namespaces")
            if label_selector:
                cmd.extend(["-l", label_selector])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"kubectl failed: {result.stderr}")
                return certs
            
            secrets = json.loads(result.stdout)
            
            for secret in secrets.get("items", []):
                if secret.get("type") != "kubernetes.io/tls":
                    continue
                
                secret_name = secret["metadata"]["name"]
                secret_ns = secret["metadata"].get("namespace", "default")
                
                # Get certificate data
                cert_b64 = secret.get("data", {}).get("tls.crt")
                if not cert_b64:
                    continue
                
                try:
                    cert_pem = base64.b64decode(cert_b64).decode()
                    parsed = self._parse_pem_certificate(cert_pem)
                    if parsed:
                        parsed.source = "kubernetes"
                        parsed.source_path = f"{secret_ns}/{secret_name}"
                        certs.append(parsed)
                except Exception as e:
                    logger.warning(f"Failed to parse cert from {secret_ns}/{secret_name}: {e}")
                    
        except FileNotFoundError:
            logger.warning("kubectl not found, skipping Kubernetes scan")
        except Exception as e:
            logger.error(f"Kubernetes scan failed: {e}")
        
        return certs
    
    # === Nginx Scanning ===
    
    def scan_nginx(
        self,
        config_paths: Optional[List[str]] = None,
    ) -> List[DiscoveredCertificate]:
        """
        Scan Nginx configurations for SSL certificates.
        
        Parses ssl_certificate directives.
        """
        if config_paths is None:
            config_paths = [
                "/etc/nginx/nginx.conf",
                "/etc/nginx/conf.d",
                "/etc/nginx/sites-enabled",
            ]
        
        certs = []
        cert_files = set()
        
        for path in config_paths:
            path = Path(path)
            if not path.exists():
                continue
            
            if path.is_file():
                cert_files.update(self._extract_nginx_certs(path))
            elif path.is_dir():
                for conf in path.glob("**/*.conf"):
                    cert_files.update(self._extract_nginx_certs(conf))
        
        for cert_file in cert_files:
            try:
                cert_path = Path(cert_file)
                if not cert_path.exists():
                    continue
                
                pem_data = cert_path.read_text()
                parsed = self._parse_pem_certificate(pem_data)
                if parsed:
                    parsed.source = "nginx"
                    parsed.source_path = str(cert_path)
                    certs.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse nginx cert {cert_file}: {e}")
        
        return certs
    
    def _extract_nginx_certs(self, config_file: Path) -> set:
        """Extract ssl_certificate paths from nginx config."""
        cert_files = set()
        
        try:
            content = config_file.read_text()
            import re
            
            # Match: ssl_certificate /path/to/cert.pem;
            pattern = r'ssl_certificate\s+([^;]+);'
            matches = re.findall(pattern, content)
            
            for match in matches:
                cert_path = match.strip().strip('"\'')
                if not cert_path.endswith("_key"):
                    cert_files.add(cert_path)
                    
        except Exception as e:
            logger.warning(f"Failed to parse nginx config {config_file}: {e}")
        
        return cert_files
    
    # === Envoy Scanning ===
    
    def scan_envoy(
        self,
        config_path: str = "/etc/envoy/envoy.yaml",
    ) -> List[DiscoveredCertificate]:
        """
        Scan Envoy configuration for TLS certificates.
        
        Looks for tls_certificates in listeners.
        """
        certs = []
        config_file = Path(config_path)
        
        if not config_file.exists():
            return certs
        
        try:
            import yaml
            
            content = config_file.read_text()
            config = yaml.safe_load(content)
            
            # Navigate to TLS contexts
            for listener in config.get("static_resources", {}).get("listeners", []):
                for filter_chain in listener.get("filter_chains", []):
                    tls_context = filter_chain.get("transport_socket", {}).get("typed_config", {})
                    
                    for cert_config in tls_context.get("common_tls_context", {}).get("tls_certificates", []):
                        cert_chain = cert_config.get("certificate_chain", {})
                        
                        if "filename" in cert_chain:
                            cert_path = Path(cert_chain["filename"])
                            if cert_path.exists():
                                pem_data = cert_path.read_text()
                                parsed = self._parse_pem_certificate(pem_data)
                                if parsed:
                                    parsed.source = "envoy"
                                    parsed.source_path = str(cert_path)
                                    certs.append(parsed)
                                    
        except ImportError:
            logger.warning("PyYAML not installed, skipping Envoy scan")
        except Exception as e:
            logger.error(f"Envoy scan failed: {e}")
        
        return certs
    
    # === File System Scanning ===
    
    def scan_filesystem(
        self,
        directories: Optional[List[str]] = None,
        extensions: Optional[List[str]] = None,
    ) -> List[DiscoveredCertificate]:
        """
        Scan file system for PEM certificate files.
        """
        if directories is None:
            directories = ["/etc/ssl/certs", "/etc/pki/tls/certs"]
        
        if extensions is None:
            extensions = [".pem", ".crt", ".cer"]
        
        certs = []
        
        for dir_path in directories:
            path = Path(dir_path)
            if not path.exists():
                continue
            
            for ext in extensions:
                for cert_file in path.glob(f"**/*{ext}"):
                    try:
                        pem_data = cert_file.read_text()
                        parsed = self._parse_pem_certificate(pem_data)
                        if parsed:
                            parsed.source = "file"
                            parsed.source_path = str(cert_file)
                            certs.append(parsed)
                    except Exception as e:
                        logger.debug(f"Failed to parse {cert_file}: {e}")
        
        return certs
    
    # === Network Scanning ===
    
    def scan_network(
        self,
        host: str,
        port: int = 443,
        timeout: float = 5.0,
    ) -> Optional[DiscoveredCertificate]:
        """
        Fetch certificate from a live TLS connection.
        """
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((host, port), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    cert_der = ssock.getpeercert(binary_form=True)
                    
                    if HAS_CRYPTOGRAPHY and cert_der:
                        cert = x509.load_der_x509_certificate(cert_der, default_backend())
                        parsed = self._parse_x509_certificate(cert)
                        if parsed:
                            parsed.source = "network"
                            parsed.source_path = f"{host}:{port}"
                            return parsed
                            
        except Exception as e:
            logger.warning(f"Network scan of {host}:{port} failed: {e}")
        
        return None
    
    # === Certificate Parsing ===
    
    def _parse_pem_certificate(self, pem_data: str) -> Optional[DiscoveredCertificate]:
        """Parse a PEM-encoded certificate."""
        if not HAS_CRYPTOGRAPHY:
            return None
        
        try:
            cert = x509.load_pem_x509_certificate(pem_data.encode(), default_backend())
            result = self._parse_x509_certificate(cert)
            if result:
                result.pem_data = pem_data
            return result
        except Exception as e:
            logger.debug(f"Failed to parse PEM: {e}")
            return None
    
    def _parse_x509_certificate(self, cert: Any) -> Optional[DiscoveredCertificate]:
        """Parse a cryptography x509 certificate object."""
        if not HAS_CRYPTOGRAPHY:
            return None
        
        try:
            # Fingerprint
            fingerprint = cert.fingerprint(cert.signature_hash_algorithm).hex()
            
            # Subject and issuer
            subject_dn = cert.subject.rfc4514_string()
            issuer_dn = cert.issuer.rfc4514_string()
            
            # SANs
            sans = []
            try:
                san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
                for name in san_ext.value:
                    if isinstance(name, x509.DNSName):
                        sans.append(name.value)
                    elif isinstance(name, x509.IPAddress):
                        sans.append(str(name.value))
            except x509.ExtensionNotFound:
                pass
            
            # Key info
            public_key = cert.public_key()
            key_type = type(public_key).__name__.replace("PublicKey", "").replace("_", "")
            try:
                key_size = public_key.key_size
            except AttributeError:
                key_size = 0
            
            # EKU
            eku_server = False
            eku_client = False
            try:
                eku_ext = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
                for usage in eku_ext.value:
                    if usage == x509.oid.ExtendedKeyUsageOID.SERVER_AUTH:
                        eku_server = True
                    elif usage == x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH:
                        eku_client = True
            except x509.ExtensionNotFound:
                pass
            
            return DiscoveredCertificate(
                source="",
                source_path="",
                fingerprint_sha256=fingerprint,
                serial_number=hex(cert.serial_number),
                subject_dn=subject_dn,
                issuer_dn=issuer_dn,
                sans=sans,
                not_before=cert.not_valid_before,
                not_after=cert.not_valid_after,
                key_type=key_type,
                key_size=key_size,
                signature_algorithm=cert.signature_algorithm_oid._name,
                eku_server_auth=eku_server,
                eku_client_auth=eku_client,
                is_self_signed=(subject_dn == issuer_dn),
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse x509: {e}")
            return None
    
    # === Unified Scan ===
    
    def scan_all(
        self,
        include_kubernetes: bool = True,
        include_nginx: bool = True,
        include_envoy: bool = True,
        include_filesystem: bool = False,
    ) -> List[DiscoveredCertificate]:
        """
        Run all scanners and return combined results.
        """
        all_certs = []
        
        if include_kubernetes:
            logger.info("Scanning Kubernetes...")
            all_certs.extend(self.scan_kubernetes())
        
        if include_nginx:
            logger.info("Scanning Nginx...")
            all_certs.extend(self.scan_nginx())
        
        if include_envoy:
            logger.info("Scanning Envoy...")
            all_certs.extend(self.scan_envoy())
        
        if include_filesystem:
            logger.info("Scanning filesystem...")
            all_certs.extend(self.scan_filesystem())
        
        logger.info(f"Total certificates discovered: {len(all_certs)}")
        return all_certs
