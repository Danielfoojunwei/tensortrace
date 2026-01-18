import logging
import time
import os
import json
from typing import List, Dict, Any
from .client import IdentityAgentClient
from .csr_generator import CSRGenerator
from .deployers import DeployerFactory
from ...utils.production_gates import is_production, ProductionGateError

logger = logging.getLogger(__name__)

class WorkPoller:
    """
    Polls the Control Plane for identity renewal jobs and executes them.
    """
    def __init__(self, config, fleet_id: str, api_key: str, csr_generator: CSRGenerator):
        self.config = config
        self.client = IdentityAgentClient(config.platform_url, fleet_id, api_key)
        self.csr_generator = csr_generator
        self.running = False
        self._job_key_map_path = os.path.join(self.config.data_dir, "identity_job_keys.json")

    def _load_job_key_map(self) -> Dict[str, str]:
        try:
            with open(self._job_key_map_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning("Failed to read job key map: %s", exc)
            return {}

    def _save_job_key_map(self, mapping: Dict[str, str]) -> None:
        os.makedirs(os.path.dirname(self._job_key_map_path), exist_ok=True)
        with open(self._job_key_map_path, "w", encoding="utf-8") as handle:
            json.dump(mapping, handle)
        os.chmod(self._job_key_map_path, 0o600)

    def _record_job_key(self, job_id: str, key_id: str) -> None:
        mapping = self._load_job_key_map()
        mapping[job_id] = key_id
        self._save_job_key_map(mapping)

    def _get_job_key(self, job_id: str) -> str:
        mapping = self._load_job_key_map()
        key_id = mapping.get(job_id)
        if not key_id:
            raise ValueError(f"Missing key mapping for job {job_id}")
        return key_id

    def _resolve_acme_webroot(self) -> str:
        env_webroot = os.getenv("TG_ACME_WEBROOT")
        if env_webroot:
            return env_webroot

        fallback = os.path.join(self.config.data_dir, "acme-challenges")
        if is_production():
            raise ProductionGateError(
                gate_name="ACME_WEBROOT",
                message="ACME webroot is required for HTTP-01 challenges in production.",
                remediation="Set TG_ACME_WEBROOT to the webroot served by your edge ingress.",
            )
        return fallback

    def poll_and_execute(self):
        """Single poll and execution cycle."""
        try:
            # 1. Get pending jobs for this fleet
            jobs = self.client.signed_request("GET", "/api/v1/identity/agent/jobs")
            
            for job in jobs:
                try:
                    self._process_job(job)
                except Exception as e:
                    logger.error(f"Failed to process job {job.get('id')}: {e}")
                    
        except Exception as e:
            logger.error(f"WorkPoller poll error: {e}")

    def _process_job(self, job: Dict[str, Any]):
        job_id = job["id"]
        status = job["status"]
        
        if status == "csr_requested":
            self._handle_csr_request(job)
        elif status == "challenge_pending":
            self._handle_challenge(job)
        elif status == "issued":
            self._handle_deployment(job)
        else:
            logger.debug(f"Job {job_id} in status {status}, no agent action required")

    def _handle_csr_request(self, job: Dict[str, Any]):
        job_id = job["id"]
        endpoint_id = job["endpoint_id"]
        
        logger.info(f"Generating CSR for renewal job {job_id}")
        
        # In a real system, we'd fetch endpoint details or SANs from the job or config
        # For now, we use the hostname from the job if available, or a default
        common_name = job.get("hostname", "managed-endpoint.local")
        if job.get("endpoint") and job["endpoint"].get("hostname"):
            common_name = job["endpoint"]["hostname"]
        
        # 1. Generate CSR with new key
        result = self.csr_generator.generate_csr_with_new_key(
            common_name=common_name,
            sans=[common_name],
            key_type="RSA", # Default
            key_size=2048,
        )
        self._record_job_key(job_id, result.key_id)
        
        # 2. Submit CSR to platform
        payload = {
            "job_id": job_id,
            "csr_pem": result.csr_pem
        }
        self.client.signed_request("POST", "/api/v1/identity/agent/csr", json_data=payload)
        logger.info(f"Submitted CSR for job {job_id}")

    def _handle_challenge(self, job: Dict[str, Any]):
        job_id = job["id"]
        challenge_type = job.get("challenge_type")
        token = job.get("challenge_token")
        
        logger.info(f"Handling {challenge_type} challenge for job {job_id}")
        
        if challenge_type == "http-01":
            # For HTTP-01, we need to serve the token at /.well-known/acme-challenge/<token>
            # In production: write to webroot or update ingress/envoy
            if not token:
                raise ValueError("Missing ACME challenge token")

            webroot = self._resolve_acme_webroot()
            challenge_dir = os.path.join(webroot, ".well-known", "acme-challenge")
            os.makedirs(challenge_dir, exist_ok=True)
            challenge_path = os.path.join(challenge_dir, token)
            with open(challenge_path, "w", encoding="utf-8") as handle:
                handle.write(token)
            logger.info("Wrote ACME challenge token to %s", challenge_path)
            
        # Notify platform that challenge is "complete" (ready for verification)
        payload = {
            "job_id": job_id,
            "token": token
        }
        self.client.signed_request("POST", "/api/v1/identity/agent/challenge-complete", json_data=payload)
        logger.info(f"Confirmed challenge completion for job {job_id}")

    def _handle_deployment(self, job: Dict[str, Any]):
        job_id = job["id"]
        endpoint_id = job["endpoint_id"]
        cert_pem = job.get("issued_cert_pem")
        endpoint = job.get("endpoint")

        # We need the private key associated with this renewal
        # In this simplistic MVP, we'll assume the most recently generated key for this endpoint
        # In production, we'd track key_id in the renewal job
        
        logger.info(f"Deploying issued certificate for job {job_id}")
        
        if not cert_pem:
            raise ValueError("Issued certificate missing for deployment")

        if not endpoint:
            raise ValueError("Endpoint metadata missing for deployment")

        key_id = self._get_job_key(job_id)
        key_pem = self.csr_generator.export_private_key_pem(key_id)

        deployer = DeployerFactory.get_deployer(endpoint["endpoint_type"])
        if endpoint["endpoint_type"] == "kubernetes":
            if not endpoint.get("k8s_namespace") or not endpoint.get("k8s_secret_name"):
                raise ValueError("Missing Kubernetes namespace or secret name for deployment")
            result = deployer.deploy(
                namespace=endpoint["k8s_namespace"],
                secret_name=endpoint["k8s_secret_name"],
                cert_pem=cert_pem,
                key_pem=key_pem,
            )
        elif endpoint["endpoint_type"] == "nginx":
            result = deployer.deploy(
                site_name=endpoint["hostname"],
                cert_pem=cert_pem,
                key_pem=key_pem,
            )
        elif endpoint["endpoint_type"] == "envoy":
            result = deployer.deploy(
                listener_name=endpoint["hostname"],
                cert_pem=cert_pem,
                key_pem=key_pem,
            )
        else:
            raise ValueError(f"Unsupported endpoint type: {endpoint['endpoint_type']}")

        if not result.success:
            raise RuntimeError(f"Deployment failed: {result.message}")

        payload = {
            "job_id": job_id
        }
        self.client.signed_request("POST", "/api/v1/identity/agent/deploy-confirm", json_data=payload)
        logger.info(f"Confirmed deployment for job {job_id}")
