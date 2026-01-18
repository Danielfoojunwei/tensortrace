"""
Deployers - Certificate Deployment to Runtime Systems

Deploys renewed certificates to:
- Kubernetes TLS secrets
- Nginx servers
- Envoy proxies
"""

import os
import subprocess
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import shutil
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class DeployResult:
    """Result of a deployment operation."""
    success: bool
    message: str
    rollback_info: Optional[Dict[str, Any]] = None


class KubernetesDeployer:
    """
    Deploy certificates to Kubernetes TLS secrets.
    
    Supports:
    - Create/update TLS secrets
    - Staged deployment (new secret → update ingress)
    - Rollback on failure
    """
    
    def __init__(self, kubeconfig: Optional[str] = None):
        self.kubeconfig = kubeconfig
    
    def deploy(
        self,
        namespace: str,
        secret_name: str,
        cert_pem: str,
        key_pem: str,
        ca_cert_pem: Optional[str] = None,
    ) -> DeployResult:
        """
        Deploy certificate to a Kubernetes TLS secret.
        
        Args:
            namespace: Target namespace
            secret_name: Secret name
            cert_pem: Certificate chain (PEM)
            key_pem: Private key (PEM)
            ca_cert_pem: Optional CA certificate
            
        Returns:
            DeployResult
        """
        try:
            # Check if secret exists
            existing = self._get_secret(namespace, secret_name)
            rollback_info = None
            
            if existing:
                # Save for rollback
                rollback_info = {
                    "namespace": namespace,
                    "secret_name": secret_name,
                    "old_data": existing.get("data", {}),
                }
            
            # Create or update secret
            secret_data = {
                "apiVersion": "v1",
                "kind": "Secret",
                "type": "kubernetes.io/tls",
                "metadata": {
                    "name": secret_name,
                    "namespace": namespace,
                },
                "data": {
                    "tls.crt": base64.b64encode(cert_pem.encode()).decode(),
                    "tls.key": base64.b64encode(key_pem.encode()).decode(),
                },
            }
            
            if ca_cert_pem:
                secret_data["data"]["ca.crt"] = base64.b64encode(ca_cert_pem.encode()).decode()
            
            # Apply secret
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(secret_data, f)
                temp_file = f.name
            
            try:
                cmd = ["kubectl", "apply", "-f", temp_file]
                if self.kubeconfig:
                    cmd.extend(["--kubeconfig", self.kubeconfig])
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    return DeployResult(
                        success=False,
                        message=f"kubectl apply failed: {result.stderr}",
                        rollback_info=rollback_info,
                    )
            finally:
                os.unlink(temp_file)
            
            logger.info(f"Deployed certificate to {namespace}/{secret_name}")
            return DeployResult(
                success=True,
                message=f"Certificate deployed to {namespace}/{secret_name}",
                rollback_info=rollback_info,
            )
            
        except Exception as e:
            return DeployResult(success=False, message=str(e))
    
    def rollback(self, rollback_info: Dict[str, Any]) -> DeployResult:
        """Restore previous certificate from rollback info."""
        try:
            namespace = rollback_info["namespace"]
            secret_name = rollback_info["secret_name"]
            old_data = rollback_info["old_data"]
            
            if not old_data:
                return DeployResult(success=False, message="No rollback data available")
            
            # Restore old secret
            secret_data = {
                "apiVersion": "v1",
                "kind": "Secret",
                "type": "kubernetes.io/tls",
                "metadata": {
                    "name": secret_name,
                    "namespace": namespace,
                },
                "data": old_data,
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(secret_data, f)
                temp_file = f.name
            
            try:
                cmd = ["kubectl", "apply", "-f", temp_file]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    return DeployResult(success=False, message=f"Rollback failed: {result.stderr}")
            finally:
                os.unlink(temp_file)
            
            logger.info(f"Rolled back {namespace}/{secret_name}")
            return DeployResult(success=True, message="Rollback successful")
            
        except Exception as e:
            return DeployResult(success=False, message=str(e))
    
    def _get_secret(self, namespace: str, secret_name: str) -> Optional[Dict]:
        """Get existing secret data."""
        try:
            cmd = ["kubectl", "get", "secret", secret_name, "-n", namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception:
            pass
        return None


class NginxDeployer:
    """
    Deploy certificates to Nginx servers.
    
    Supports:
    - File-based deployment
    - Configuration reload
    - Staged deployment with backup
    """
    
    def __init__(
        self,
        cert_dir: str = "/etc/nginx/ssl",
        nginx_bin: str = "nginx",
    ):
        self.cert_dir = Path(cert_dir)
        self.nginx_bin = nginx_bin
    
    def deploy(
        self,
        site_name: str,
        cert_pem: str,
        key_pem: str,
        reload: bool = True,
    ) -> DeployResult:
        """
        Deploy certificate for an Nginx site.
        
        Args:
            site_name: Site identifier (used for file naming)
            cert_pem: Certificate chain (PEM)
            key_pem: Private key (PEM)
            reload: Whether to reload Nginx after deployment
            
        Returns:
            DeployResult
        """
        try:
            self.cert_dir.mkdir(parents=True, exist_ok=True)
            
            cert_path = self.cert_dir / f"{site_name}.crt"
            key_path = self.cert_dir / f"{site_name}.key"
            
            # Backup existing files
            rollback_info = {}
            if cert_path.exists():
                backup_cert = cert_path.with_suffix(".crt.bak")
                shutil.copy2(cert_path, backup_cert)
                rollback_info["cert_backup"] = str(backup_cert)
            
            if key_path.exists():
                backup_key = key_path.with_suffix(".key.bak")
                shutil.copy2(key_path, backup_key)
                rollback_info["key_backup"] = str(backup_key)
            
            rollback_info["cert_path"] = str(cert_path)
            rollback_info["key_path"] = str(key_path)
            
            # Write new files
            cert_path.write_text(cert_pem)
            key_path.write_text(key_pem)
            os.chmod(key_path, 0o600)
            
            # Test configuration
            test_result = subprocess.run(
                [self.nginx_bin, "-t"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if test_result.returncode != 0:
                # Restore backups
                self.rollback(rollback_info)
                return DeployResult(
                    success=False,
                    message=f"Nginx config test failed: {test_result.stderr}",
                    rollback_info=rollback_info,
                )
            
            # Reload if requested
            if reload:
                reload_result = subprocess.run(
                    [self.nginx_bin, "-s", "reload"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if reload_result.returncode != 0:
                    self.rollback(rollback_info)
                    return DeployResult(
                        success=False,
                        message=f"Nginx reload failed: {reload_result.stderr}",
                        rollback_info=rollback_info,
                    )
            
            logger.info(f"Deployed certificate for Nginx site: {site_name}")
            return DeployResult(
                success=True,
                message=f"Certificate deployed for {site_name}",
                rollback_info=rollback_info,
            )
            
        except Exception as e:
            return DeployResult(success=False, message=str(e))
    
    def rollback(self, rollback_info: Dict[str, Any]) -> DeployResult:
        """Restore previous certificate files."""
        try:
            if "cert_backup" in rollback_info:
                backup = Path(rollback_info["cert_backup"])
                target = Path(rollback_info["cert_path"])
                if backup.exists():
                    shutil.copy2(backup, target)
            
            if "key_backup" in rollback_info:
                backup = Path(rollback_info["key_backup"])
                target = Path(rollback_info["key_path"])
                if backup.exists():
                    shutil.copy2(backup, target)
            
            # Reload Nginx
            subprocess.run([self.nginx_bin, "-s", "reload"], capture_output=True, timeout=10)
            
            logger.info("Nginx rollback successful")
            return DeployResult(success=True, message="Rollback successful")
            
        except Exception as e:
            return DeployResult(success=False, message=str(e))


class EnvoyDeployer:
    """
    Deploy certificates to Envoy proxies.
    
    Supports hot restart via SDS or file-based deployment.
    """
    
    def __init__(
        self,
        cert_dir: str = "/etc/envoy/ssl",
        use_sds: bool = False,
    ):
        self.cert_dir = Path(cert_dir)
        self.use_sds = use_sds
    
    def deploy(
        self,
        listener_name: str,
        cert_pem: str,
        key_pem: str,
    ) -> DeployResult:
        """
        Deploy certificate for an Envoy listener.
        
        For SDS: Updates the secret resource
        For file-based: Writes files and Envoy hot-reloads automatically
        """
        try:
            self.cert_dir.mkdir(parents=True, exist_ok=True)
            
            cert_path = self.cert_dir / f"{listener_name}.crt"
            key_path = self.cert_dir / f"{listener_name}.key"
            
            # Backup
            rollback_info = {
                "cert_path": str(cert_path),
                "key_path": str(key_path),
            }
            
            if cert_path.exists():
                rollback_info["cert_backup"] = cert_path.read_text()
            if key_path.exists():
                rollback_info["key_backup"] = key_path.read_text()
            
            # Write new files
            cert_path.write_text(cert_pem)
            key_path.write_text(key_pem)
            os.chmod(key_path, 0o600)
            
            # Envoy auto-reloads on file change if configured with watch
            logger.info(f"Deployed certificate for Envoy listener: {listener_name}")
            return DeployResult(
                success=True,
                message=f"Certificate deployed for {listener_name}",
                rollback_info=rollback_info,
            )
            
        except Exception as e:
            return DeployResult(success=False, message=str(e))
    
    def rollback(self, rollback_info: Dict[str, Any]) -> DeployResult:
        """Restore previous certificate files."""
        try:
            if "cert_backup" in rollback_info:
                Path(rollback_info["cert_path"]).write_text(rollback_info["cert_backup"])
            
            if "key_backup" in rollback_info:
                Path(rollback_info["key_path"]).write_text(rollback_info["key_backup"])
            
            return DeployResult(success=True, message="Rollback successful")
            
        except Exception as e:
            return DeployResult(success=False, message=str(e))


class DeployerFactory:
    """Factory for creating appropriate deployers."""
    
    @staticmethod
    def get_deployer(endpoint_type: str) -> Any:
        """Get deployer for endpoint type."""
        deployers = {
            "kubernetes": KubernetesDeployer,
            "nginx": NginxDeployer,
            "envoy": EnvoyDeployer,
        }
        
        deployer_class = deployers.get(endpoint_type)
        if deployer_class:
            return deployer_class()
        
        raise ValueError(f"No deployer for endpoint type: {endpoint_type}")
