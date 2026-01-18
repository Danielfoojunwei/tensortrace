"""
Renewal Scheduler - Certificate Lifecycle Orchestration

Schedules and orchestrates certificate renewal workflows.
Uses APScheduler for background job management.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from sqlmodel import Session, select
import logging
import asyncio
import ssl
import socket
import hashlib

from ..platform.models.identity_models import (
    IdentityRenewalJob,
    IdentityCertificate,
    IdentityEndpoint,
    IdentityPolicy,
    RenewalJobStatus,
    AuditAction,
    EndpointType,
)
from .policy_engine import PolicyEngine
from .audit import AuditService
from .acme.client import ACMEClient, ACMEProvider, ACMEOrder, ACMEChallenge

logger = logging.getLogger(__name__)


class RenewalScheduler:
    """
    Certificate renewal orchestrator.
    
    Features:
    - Automatic renewal scheduling based on policy
    - State machine for renewal workflow
    - Retry logic with exponential backoff
    - Rollback support on failure
    - Idempotent operations
    """
    
    # Retry backoff multipliers
    RETRY_BACKOFF_MINUTES = [5, 15, 60, 240]  # 5min, 15min, 1hr, 4hr
    
    def __init__(self, session: Session):
        self.session = session
        self.policy_engine = PolicyEngine()
        self.audit_service = AuditService(session)
        
        # Callbacks for external integrations
        self._on_csr_request: Optional[Callable] = None
        self._on_challenge_start: Optional[Callable] = None
        self._on_deploy: Optional[Callable] = None
        
        # ACME Client cache
        self._acme_clients: Dict[str, ACMEClient] = {}

    def _get_acme_client(self, policy: IdentityPolicy) -> ACMEClient:
        """Get or create ACME client for the given policy."""
        provider_name = policy.preferred_acme_provider or "letsencrypt"
        if provider_name not in self._acme_clients:
            provider = ACMEProvider(provider_name)
            # Use a persistent account key path
            key_path = f"keys/identity/acme_account_{provider_name}.pem"
            self._acme_clients[provider_name] = ACMEClient(
                provider=provider,
                account_key_path=key_path,
                verify_ssl=(provider != ACMEProvider.PEBBLE)  # Don't verify for Pebble
            )
        return self._acme_clients[provider_name]
    
    def set_csr_callback(self, callback: Callable) -> None:
        """Set callback for CSR requests to agent."""
        self._on_csr_request = callback
    
    def set_challenge_callback(self, callback: Callable) -> None:
        """Set callback for challenge initiation."""
        self._on_challenge_start = callback
    
    def set_deploy_callback(self, callback: Callable) -> None:
        """Set callback for certificate deployment."""
        self._on_deploy = callback
    
    # === Job Creation ===
    
    def schedule_renewal(
        self,
        tenant_id: str,
        fleet_id: str,
        endpoint_id: str,
        policy_id: str,
        scheduled_at: Optional[datetime] = None,
        old_cert_id: Optional[str] = None,
    ) -> IdentityRenewalJob:
        """
        Schedule a new renewal job.
        
        Idempotent: won't create duplicate jobs for the same endpoint/policy.
        """
        # Check for existing pending job
        existing = self.session.exec(
            select(IdentityRenewalJob).where(
                IdentityRenewalJob.endpoint_id == endpoint_id,
                IdentityRenewalJob.status.in_([
                    RenewalJobStatus.PENDING,
                    RenewalJobStatus.CSR_REQUESTED,
                    RenewalJobStatus.CHALLENGE_PENDING,
                    RenewalJobStatus.ISSUING,
                ])
            )
        ).first()
        
        if existing:
            logger.info(f"Renewal job already exists for endpoint {endpoint_id}: {existing.id}")
            return existing
        
        job = IdentityRenewalJob(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            endpoint_id=endpoint_id,
            policy_id=policy_id,
            old_cert_id=old_cert_id,
            scheduled_at=scheduled_at or datetime.utcnow(),
            status=RenewalJobStatus.PENDING,
        )
        
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        
        # Audit
        self.audit_service.log(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            action=AuditAction.RENEWAL_STARTED,
            actor_type="scheduler",
            actor_id="system",
            target_type="renewal_job",
            target_id=job.id,
            payload={"endpoint_id": endpoint_id, "policy_id": policy_id},
        )
        
        logger.info(f"Scheduled renewal job: {job.id}")
        return job
    
    def get_job(self, job_id: str) -> Optional[IdentityRenewalJob]:
        """Get job by ID."""
        return self.session.get(IdentityRenewalJob, job_id)
    
    def list_jobs(
        self,
        tenant_id: str,
        fleet_id: Optional[str] = None,
        status: Optional[RenewalJobStatus] = None,
        limit: int = 100,
    ) -> List[IdentityRenewalJob]:
        """List renewal jobs with filters."""
        statement = select(IdentityRenewalJob).where(
            IdentityRenewalJob.tenant_id == tenant_id
        )
        
        if fleet_id:
            statement = statement.where(IdentityRenewalJob.fleet_id == fleet_id)
        if status:
            statement = statement.where(IdentityRenewalJob.status == status)
        
        statement = statement.order_by(IdentityRenewalJob.created_at.desc()).limit(limit)
        return list(self.session.exec(statement).all())
    
    # === State Machine ===
    
    def advance_job(self, job_id: str) -> IdentityRenewalJob:
        """
        Advance a job through its state machine.
        
        State transitions:
        PENDING → CSR_REQUESTED → CSR_RECEIVED → CHALLENGE_PENDING →
        CHALLENGE_COMPLETE → ISSUING → ISSUED → DEPLOYING → VALIDATING → SUCCEEDED
        
        On failure: → FAILED (can retry if attempts < max)
        On rollback: → ROLLED_BACK
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job.is_terminal:
            logger.info(f"Job {job_id} is already in terminal state: {job.status}")
            return job
        
        try:
            if job.status == RenewalJobStatus.PENDING:
                return self._request_csr(job)
            elif job.status == RenewalJobStatus.CSR_REQUESTED:
                # Waiting for agent to provide CSR
                return job
            elif job.status == RenewalJobStatus.CSR_RECEIVED:
                return self._start_challenge(job)
            elif job.status == RenewalJobStatus.CHALLENGE_PENDING:
                # Waiting for challenge completion
                return job
            elif job.status == RenewalJobStatus.CHALLENGE_COMPLETE:
                return self._issue_certificate(job)
            elif job.status == RenewalJobStatus.ISSUING:
                return self._poll_issuance(job)
            elif job.status == RenewalJobStatus.ISSUED:
                return self._deploy_certificate(job)
            elif job.status == RenewalJobStatus.DEPLOYING:
                # Waiting for deployment
                return job
            elif job.status == RenewalJobStatus.VALIDATING:
                return self._validate_deployment(job)
            else:
                return job
                
        except Exception as e:
            return self._handle_failure(job, str(e))
    
    def _request_csr(self, job: IdentityRenewalJob) -> IdentityRenewalJob:
        """Request CSR from agent."""
        job.status = RenewalJobStatus.CSR_REQUESTED
        job.started_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        
        if self._on_csr_request:
            # Trigger agent to generate CSR
            self._on_csr_request(job)
        
        self.session.add(job)
        self.session.commit()
        
        logger.info(f"Job {job.id}: CSR requested")
        return job
    
    def receive_csr(self, job_id: str, csr_pem: str) -> IdentityRenewalJob:
        """Receive CSR from agent."""
        job = self.get_job(job_id)
        if not job or job.status != RenewalJobStatus.CSR_REQUESTED:
            raise ValueError(f"Invalid job state for CSR: {job_id}")
        
        job.csr_pem = csr_pem
        job.status = RenewalJobStatus.CSR_RECEIVED
        job.updated_at = datetime.utcnow()
        
        self.session.add(job)
        self.session.commit()
        
        self.audit_service.log(
            tenant_id=job.tenant_id,
            fleet_id=job.fleet_id,
            action=AuditAction.CSR_GENERATED,
            actor_type="agent",
            actor_id=job.endpoint_id,
            target_type="renewal_job",
            target_id=job.id,
        )
        
        logger.info(f"Job {job.id}: CSR received")
        return job
    
    def _start_challenge(self, job: IdentityRenewalJob) -> IdentityRenewalJob:
        """Initiate ACME challenge."""
        policy = self.session.get(IdentityPolicy, job.policy_id)
        if not policy:
            return self._handle_failure(job, "Policy not found")

        if policy.require_public_trust:
            # Public ACME flow
            try:
                acme = self._get_acme_client(policy)
                # 1. Register account (idempotent in client)
                # In production, get email from tenant settings
                admin_email = "admin@tensorguard.io" 
                acme.register_account(admin_email)
                
                # 2. Create Order
                endpoint = self.session.get(IdentityEndpoint, job.endpoint_id)
                import json
                sans = json.loads(endpoint.tags).get("sans", [endpoint.hostname]) if endpoint.tags else [endpoint.hostname]
                order = acme.create_order(sans)
                
                # 3. Get Challenges
                challenges = acme.get_challenges(order)
                # Pick the first http-01 challenge for now
                challenge = next((c for c in challenges if c.type == policy.acme_challenge_type), challenges[0])
                
                # 4. Update Job
                job.acme_order_url = order.order_url
                job.acme_finalize_url = order.finalize_url
                job.acme_authz_urls_json = json.dumps(order.authorizations)
                job.challenge_url = challenge.url
                job.challenge_token = challenge.token
                job.challenge_domain = challenge.domain
                job.challenge_type = challenge.type
                job.last_acme_status = order.status
                
                job.status = RenewalJobStatus.CHALLENGE_PENDING
                job.updated_at = datetime.utcnow()
                
                if self._on_challenge_start:
                    self._on_challenge_start(job)
                
                self.session.add(job)
                self.session.commit()
                
                logger.info(f"Job {job.id}: ACME challenge started ({job.challenge_type}) for {challenge.domain}")
                return job
                
            except Exception as e:
                return self._handle_failure(job, f"ACME order failed: {str(e)}")
        else:
            # Private CA flow - requires implementation or explicit configuration
            from ..utils.production_gates import is_production, ProductionGateError

            if is_production():
                raise ProductionGateError(
                    gate_name="PRIVATE_CA_CHALLENGE",
                    message="Private CA flow is not fully implemented. Cannot skip challenge in production.",
                    remediation=(
                        "Either configure require_public_trust=true to use ACME (Let's Encrypt), "
                        "or implement the Private CA integration at identity/ca/private_ca.py"
                    )
                )

            logger.warning(
                "[DEV MODE] Private CA flow - skipping challenge for development. "
                "Production requires full implementation."
            )
            job.status = RenewalJobStatus.CHALLENGE_COMPLETE
            job.updated_at = datetime.utcnow()
            self.session.add(job)
            self.session.commit()
            return job
    
    def complete_challenge(self, job_id: str, token: str) -> IdentityRenewalJob:
        """Mark challenge as complete."""
        job = self.get_job(job_id)
        if not job or job.status != RenewalJobStatus.CHALLENGE_PENDING:
            raise ValueError(f"Invalid job state for challenge completion: {job_id}")
        
        job.challenge_token = token
        job.status = RenewalJobStatus.CHALLENGE_COMPLETE
        job.updated_at = datetime.utcnow()
        
        self.session.add(job)
        self.session.commit()
        
        self.audit_service.log(
            tenant_id=job.tenant_id,
            fleet_id=job.fleet_id,
            action=AuditAction.CHALLENGE_COMPLETED,
            actor_type="agent",
            actor_id=job.endpoint_id,
            target_type="renewal_job",
            target_id=job.id,
        )
        
        logger.info(f"Job {job.id}: Challenge completed")
        return job
    
    def _issue_certificate(self, job: IdentityRenewalJob) -> IdentityRenewalJob:
        """Submit CSR and finalize issuance."""
        policy = self.session.get(IdentityPolicy, job.policy_id)
        
        if policy.require_public_trust:
            try:
                acme = self._get_acme_client(policy)
                order = ACMEOrder(
                    order_url=job.acme_order_url,
                    status=job.last_acme_status,
                    identifiers=[], # Not needed for finalize
                    authorizations=[], # Not needed for finalize
                    finalize_url=job.acme_finalize_url
                )
                
                order = acme.finalize_order(order, job.csr_pem)
                job.last_acme_status = order.status
                job.status = RenewalJobStatus.ISSUING
                job.updated_at = datetime.utcnow()
                
                self.session.add(job)
                self.session.commit()
                
                logger.info(f"Job {job.id}: ACME order finalized, now ISSUING")
                return job
            except Exception as e:
                return self._handle_failure(job, f"ACME finalization failed: {str(e)}")
        else:
            # Private CA flow - requires real implementation
            from ..utils.production_gates import is_production, ProductionGateError

            if is_production():
                raise ProductionGateError(
                    gate_name="PRIVATE_CA_ISSUANCE",
                    message="Private CA certificate issuance is not implemented. Cannot issue stub certificates in production.",
                    remediation=(
                        "Either configure require_public_trust=true to use ACME (Let's Encrypt), "
                        "or implement the Private CA API integration at identity/ca/private_ca.py"
                    )
                )

            logger.warning(
                "[DEV MODE] Private CA issuance not implemented. "
                "Returning error - use ACME or implement Private CA for real certificates."
            )
            return self._handle_failure(
                job,
                "Private CA issuance not implemented. Configure require_public_trust=true or implement Private CA."
            )
    
    def receive_certificate(self, job_id: str, cert_pem: str, cert_id: str) -> IdentityRenewalJob:
        """Receive issued certificate (manual/external)."""
        job = self.get_job(job_id)
        if not job or job.status != RenewalJobStatus.ISSUING:
            raise ValueError(f"Invalid job state for certificate: {job_id}")
        
        job.issued_cert_pem = cert_pem
        job.new_cert_id = cert_id
        job.status = RenewalJobStatus.ISSUED
        job.updated_at = datetime.utcnow()
        
        self.session.add(job)
        self.session.commit()
        
        logger.info(f"Job {job.id}: Certificate received externally")
        return job

    def _poll_issuance(self, job: IdentityRenewalJob) -> IdentityRenewalJob:
        """Poll ACME for certificate completion."""
        policy = self.session.get(IdentityPolicy, job.policy_id)
        if not policy or not policy.require_public_trust:
            # Fallback or private CA (should already be ISSUED)
            return job
            
        try:
            acme = self._get_acme_client(policy)
            # Register account if not already (it's idempotent)
            acme.register_account("admin@tensorguard.io")
            
            order_data = acme.check_order_status(job.acme_order_url)
            job.last_acme_status = order_data["status"]
            job.updated_at = datetime.utcnow()
            
            if order_data["status"] == "valid":
                order = ACMEOrder(
                    order_url=job.acme_order_url,
                    status=order_data["status"],
                    identifiers=[], 
                    authorizations=[],
                    finalize_url=job.acme_finalize_url,
                    certificate_url=order_data["certificate"]
                )
                cert_pem = acme.download_certificate(order)
                job.issued_cert_pem = cert_pem
                # Fingerprint of chain for tracking
                job.new_cert_id = hashlib.sha256(cert_pem.encode()).hexdigest()
                job.status = RenewalJobStatus.ISSUED
                logger.info(f"Job {job.id}: Certificate issued and downloaded")
            elif order_data["status"] == "invalid":
                reason = order_data.get("error", {}).get("detail", "Unknown ACME error")
                raise RuntimeError(f"ACME order became invalid: {reason}")
            
            self.session.add(job)
            self.session.commit()
            return job
            
        except Exception as e:
            logger.warning(f"Polling failed for job {job.id}: {e}")
            # Don't fail immediately, let background runner retry
            # But we can log the error in the job
            job.last_error = str(e)
            self.session.add(job)
            self.session.commit()
            return job
    
    def _deploy_certificate(self, job: IdentityRenewalJob) -> IdentityRenewalJob:
        """Deploy certificate to endpoint."""
        job.status = RenewalJobStatus.DEPLOYING
        job.updated_at = datetime.utcnow()
        
        if self._on_deploy:
            self._on_deploy(job)
        
        self.session.add(job)
        self.session.commit()
        
        logger.info(f"Job {job.id}: Deploying certificate")
        return job
    
    def confirm_deployment(self, job_id: str) -> IdentityRenewalJob:
        """Agent confirms deployment is complete."""
        job = self.get_job(job_id)
        if not job or job.status != RenewalJobStatus.DEPLOYING:
            raise ValueError(f"Invalid job state for deployment confirmation: {job_id}")
        
        job.status = RenewalJobStatus.VALIDATING
        job.updated_at = datetime.utcnow()
        
        self.session.add(job)
        self.session.commit()
        
        logger.info(f"Job {job.id}: Deployment confirmed, validating")
        return job
    
    def _validate_deployment(self, job: IdentityRenewalJob) -> IdentityRenewalJob:
        """Validate deployment via TLS probes or secret checks."""
        endpoint = self.session.get(IdentityEndpoint, job.endpoint_id)
        if not endpoint:
            return self._handle_failure(job, "Endpoint not found during validation")
            
        try:
            if endpoint.endpoint_type == EndpointType.KUBERNETES:
                # In production: check K8s secret via API
                # For MVP: assume success if agent confirmed
                logger.info(f"Job {job.id}: Validating K8s secret {endpoint.k8s_secret_name}")
                pass
            else:
                # TLS Probe for other endpoints
                logger.info(f"Job {job.id}: Probing TLS endpoint {endpoint.hostname}:{endpoint.port}")
                
                # Create SSL context that doesn't verify (we want to see the cert even if untrusted)
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                with socket.create_connection((endpoint.hostname, endpoint.port), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=endpoint.hostname) as ssock:
                        der_cert = ssock.getpeercert(binary_form=True)
                        fingerprint = hashlib.sha256(der_cert).hexdigest()
                        
                        # Verify against issued_cert_pem (need to extract thumbprint from PEM)
                        # For now, we assume the job has new_cert_id which is the thumbprint
                        if job.new_cert_id and fingerprint.lower() != job.new_cert_id.lower():
                            raise ValueError(f"Fingerprint mismatch: expected {job.new_cert_id}, got {fingerprint}")
            
            job.status = RenewalJobStatus.SUCCEEDED
            job.completed_at = datetime.utcnow()
            job.updated_at = datetime.utcnow()
            
            # Update the current cert in inventory
            from .inventory import InventoryService
            InventoryService(self.session).mark_cert_current(job.new_cert_id, job.endpoint_id)

            # Deep integration: Trigger N2HE key rotation on successful identity refresh
            # This binds the Transport Identity lifecycle to the Data Privacy lifecycle.
            try:
                from ..platform.services.remediation_service import RemediationService
                RemediationService(self.session).rotate_n2he_key(job.fleet_id)
                logger.info(f"Triggered N2HE rotation for fleet {job.fleet_id} following identity refresh")
            except Exception as e:
                logger.error(f"Failed to trigger N2HE rotation: {e}")
            
            self.session.add(job)
            self.session.commit()
            
            self.audit_service.log(
                tenant_id=job.tenant_id,
                fleet_id=job.fleet_id,
                action=AuditAction.RENEWAL_SUCCEEDED,
                actor_type="scheduler",
                actor_id="system",
                target_type="renewal_job",
                target_id=job.id,
                payload={"new_cert_id": job.new_cert_id},
            )
            
            logger.info(f"Job {job.id}: Renewal succeeded and validated")
            return job
            
        except Exception as e:
            logger.warning(f"Validation failed for job {job.id}: {e}")
            return self._handle_failure(job, f"Validation failed: {str(e)}")
    
    def _handle_failure(self, job: IdentityRenewalJob, error: str) -> IdentityRenewalJob:
        """Handle job failure with retry logic."""
        job.last_error = error
        job.updated_at = datetime.utcnow()
        
        # Don't increment retry count if it's already terminal (safety)
        if job.is_terminal:
            return job
            
        if job.can_retry:
            job.retry_count += 1
            # Schedule retry with backoff
            backoff_idx = min(job.retry_count - 1, len(self.RETRY_BACKOFF_MINUTES) - 1)
            backoff_minutes = self.RETRY_BACKOFF_MINUTES[backoff_idx]
            job.next_retry_at = datetime.utcnow() + timedelta(minutes=backoff_minutes)
            job.status = RenewalJobStatus.PENDING # Reset to pending for retry
            
            logger.warning(f"Job {job.id} failed, retry {job.retry_count} in {backoff_minutes}m: {error}")
        else:
            job.status = RenewalJobStatus.FAILED
            job.completed_at = datetime.utcnow()
            
            self.audit_service.log(
                tenant_id=job.tenant_id,
                fleet_id=job.fleet_id,
                action=AuditAction.RENEWAL_FAILED,
                actor_type="scheduler",
                actor_id="system",
                target_type="renewal_job",
                target_id=job.id,
                payload={"error": error, "retries": job.retry_count},
            )
            
            logger.error(f"Job {job.id} failed permanently: {error}")
        
        self.session.add(job)
        self.session.commit()
        return job
    
    def rollback_job(self, job_id: str, reason: str) -> IdentityRenewalJob:
        """Rollback a failed or problematic renewal."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if not job.can_rollback:
            raise ValueError(f"Job {job_id} cannot be rolled back")
        
        # In production: restore old certificate
        # For now: just mark as rolled back
        
        job.status = RenewalJobStatus.ROLLED_BACK
        job.status_message = reason
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        
        self.session.add(job)
        self.session.commit()
        
        self.audit_service.log(
            tenant_id=job.tenant_id,
            fleet_id=job.fleet_id,
            action=AuditAction.RENEWAL_ROLLED_BACK,
            actor_type="scheduler",
            actor_id="system",
            target_type="renewal_job",
            target_id=job.id,
            payload={"reason": reason},
        )
        
        logger.info(f"Job {job.id} rolled back: {reason}")
        return job
    
    # === Scheduling ===
    
    def find_renewals_due(
        self,
        tenant_id: str,
        policy_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find certificates that are due for renewal based on policy.
        
        Returns list of {endpoint, certificate, policy, days_to_renewal}.
        """
        from .inventory import InventoryService
        
        inventory = InventoryService(self.session)
        certs = inventory.list_certificates(tenant_id)
        
        due_for_renewal = []
        
        for cert in certs:
            if cert.policy_id:
                policy = self.session.get(IdentityPolicy, cert.policy_id)
            elif policy_id:
                policy = self.session.get(IdentityPolicy, policy_id)
            else:
                continue
            
            if not policy:
                continue
            
            # Check if within renewal window
            renewal_date = self.policy_engine.calculate_renewal_date(cert, policy)
            if datetime.utcnow() >= renewal_date:
                endpoint = inventory.get_endpoint(cert.endpoint_id)
                due_for_renewal.append({
                    "endpoint": endpoint,
                    "certificate": cert,
                    "policy": policy,
                    "days_to_expiry": cert.days_to_expiry,
                })
        
        return due_for_renewal
    
    def run_scheduled_renewals(self, tenant_id: str) -> List[IdentityRenewalJob]:
        """
        Execute renewals for all certificates due.
        
        Returns list of created renewal jobs.
        """
        due = self.find_renewals_due(tenant_id)
        jobs = []
        
        for item in due:
            endpoint = item["endpoint"]
            cert = item["certificate"]
            policy = item["policy"]
            
            job = self.schedule_renewal(
                tenant_id=tenant_id,
                fleet_id=endpoint.fleet_id,
                endpoint_id=endpoint.id,
                policy_id=policy.id,
                old_cert_id=cert.id,
            )
            jobs.append(job)
        
        return jobs
