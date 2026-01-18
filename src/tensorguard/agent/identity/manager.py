"""
Identity Manager - Subsystem Controller

Manages the lifecycle of identity services within the Unified Agent.
Wraps the Scanner, CSR Generator, and communication logic.
"""

import logging
import threading
import time
from typing import Optional, List
from datetime import datetime
from ...schemas.unified_config import IdentityConfig

from .scanner import CertificateScanner
from .csr_generator import CSRGenerator
from .deployers import DeployerFactory
from .tpm_simulator import TPMSimulator
from .work_poller import WorkPoller
from .client import IdentityAgentClient

logger = logging.getLogger(__name__)

class IdentityManager:
    """
    Subsystem controller for Machine Identity Guard.
    """
    def __init__(self, agent_config: 'AgentConfig', config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.config: IdentityConfig = agent_config.identity
        self.fleet_id = agent_config.fleet_id
        self.api_key = agent_config.api_key
        
        self.scanner = CertificateScanner()
        self.csr_generator = CSRGenerator(key_storage_path=self.config.key_storage_path)
        self.tpm = TPMSimulator() # Hardware trust root
        
        # Identity client and poller
        self.client = IdentityAgentClient(
            base_url=agent_config.control_plane_url,
            fleet_id=self.fleet_id,
            api_key=self.api_key
        )
        self.poller = WorkPoller(
            config=agent_config,
            fleet_id=self.fleet_id,
            api_key=self.api_key,
            csr_generator=self.csr_generator
        )
        
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def configure(self, new_config: IdentityConfig):
        """Update configuration on the fly."""
        logger.info("Reconfiguring Identity Manager")
        self.config = new_config

    def start(self):
        """Start background tasks."""
        if not self.config.enabled:
            return
            
        logger.info("IdentityManager starting...")
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background tasks."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_loop(self):
        """Main identity loop."""
        while self.running:
            try:
                # 1. Periodic Scan
                self.run_scan()
                
                # 2. Poll for identity work (renewals, etc.)
                self.poller.poll_and_execute()
                
                # 3. Hardware Attestation Heartbeat
                self.send_heartbeat()
                
            except Exception as e:
                logger.error(f"Identity loop error: {e}")
            
            # Sleep interval
            for _ in range(self.config.scan_interval_seconds):
                if not self.running:
                    break
                time.sleep(1)

    def run_scan(self) -> List[dict]:
        """Execute a certificate scan."""
        logger.info("Executing periodic certificate scan")
        certs = self.scanner.scan_all(
            include_kubernetes=self.config.scan_kubernetes,
            include_nginx=self.config.scan_nginx,
            include_envoy=self.config.scan_envoy,
            include_filesystem=self.config.scan_filesystem,
        )
        
        self._report_certificates(certs)
        return certs
        
    def check_renewals(self):
        """Check for certificates nearing expiry (<24h) and trigger auto-renewal."""
        # Simple renewal policy check
        certs = self.scanner.scan_filesystem(self.config.key_storage_path)
        for cert in certs:
            expiry = datetime.fromisoformat(cert['not_after'].replace("Z", ""))
            days_left = (expiry - datetime.utcnow()).days
            
            if days_left < 1:
                logger.warning(f"Certificate {cert['subject']} nearing expiry. Triggering renewal.")
                # Logic: Generate new CSR -> Request Sign -> Deploy
                # Stub for MVP, but architecture fits here
                self._renew_certificate(cert)

    def send_heartbeat(self):
        """Send TPM-signed heartbeat."""
        nonce = datetime.utcnow().isoformat()
        try:
            quote = self.tpm.get_quote(nonce)
            # In production: self.client.signed_request("POST", "/agent/heartbeat", ...)
            logger.debug(f"Generated TPM quote for heartbeat: {quote['signature_hex'][:20]}...")
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")

    def _renew_certificate(self, cert_info: dict):
        """Execute renewal workflow."""
        logger.info(f"Renewing certificate for {cert_info['subject']}")
        # 1. Generate new Key/CSR
        # 2. Call Platform API
        # 3. Write new cert to disk
        pass

    def _report_certificates(self, certs: List):
        """Send discovered certificates to control plane."""
        if not certs:
            return
            
        try:
            # Map scanner format to API format
            report_data = []
            for c in certs:
                report_data.append({
                    "fingerprint_sha256": c.get("fingerprint"),
                    "serial_number": c.get("serial"),
                    "subject_dn": c.get("subject"),
                    "issuer_dn": c.get("issuer"),
                    "sans": c.get("sans", []),
                    "not_before": c.get("not_before"),
                    "not_after": c.get("not_after"),
                    "key_type": c.get("key_type", "RSA"),
                    "key_size": c.get("key_size", 2048),
                    "signature_algorithm": c.get("sig_alg", "SHA256WithRSA"),
                    "eku_server_auth": True, # Guess/Default
                    "eku_client_auth": False,
                    "is_public_trust": True,
                })
            
            self.client.signed_request("POST", "/api/v1/identity/agent/report", json_data=report_data)
            logger.info(f"Reported {len(certs)} certificates to control plane")
        except Exception as e:
            logger.error(f"Failed to report certificates: {e}")
