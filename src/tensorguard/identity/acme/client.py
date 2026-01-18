"""
ACME Client - Certificate Issuance via ACME Protocol

Supports Let's Encrypt, ZeroSSL, Buypass, and other ACME-compliant CAs.
Implements RFC 8555 ACME protocol.
"""

import json
import time
import hashlib
import base64
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import josepy for ACME crypto
try:
    import josepy as jose
    from cryptography.hazmat.primitives.asymmetric import rsa, ec
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False
    logger.warning("josepy not installed, ACME client has limited functionality")


class ACMEProvider(str, Enum):
    """Supported ACME providers."""
    LETSENCRYPT = "letsencrypt"
    LETSENCRYPT_STAGING = "letsencrypt_staging"
    ZEROSSL = "zerossl"
    BUYPASS = "buypass"
    PEBBLE = "pebble"  # Test server


ACME_DIRECTORIES = {
    ACMEProvider.LETSENCRYPT: "https://acme-v02.api.letsencrypt.org/directory",
    ACMEProvider.LETSENCRYPT_STAGING: "https://acme-staging-v02.api.letsencrypt.org/directory",
    ACMEProvider.ZEROSSL: "https://acme.zerossl.com/v2/DV90/directory",
    ACMEProvider.BUYPASS: "https://api.buypass.com/acme/directory",
    ACMEProvider.PEBBLE: "https://localhost:14000/dir",
}


@dataclass
class ACMEOrder:
    """ACME order representation."""
    order_url: str
    status: str
    identifiers: List[Dict[str, str]]
    authorizations: List[str]
    finalize_url: str
    certificate_url: Optional[str] = None


@dataclass
class ACMEChallenge:
    """ACME challenge representation."""
    type: str  # "http-01", "dns-01", "tls-alpn-01"
    url: str
    token: str
    status: str
    key_authorization: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class ACMEResult:
    """Result of ACME operations."""
    success: bool
    message: str
    certificate_pem: Optional[str] = None
    chain_pem: Optional[str] = None


class ACMEClient:
    """
    ACME protocol client for automated certificate issuance.
    
    Implements:
    - Account registration
    - Order creation
    - HTTP-01 and DNS-01 challenges
    - Certificate finalization
    """
    
    def __init__(
        self,
        provider: ACMEProvider = ACMEProvider.LETSENCRYPT,
        account_key_path: Optional[str] = None,
        directory_url: Optional[str] = None,
        verify_ssl: bool = True,
    ):
        self.provider = provider
        self.directory_url = directory_url or ACME_DIRECTORIES.get(provider)
        self.verify_ssl = verify_ssl
        self.account_key_path = Path(account_key_path) if account_key_path else None
        
        # ACME directory endpoints
        self._directory: Optional[Dict] = None
        self._nonce: Optional[str] = None
        
        # Account
        self._account_key = None
        self._account_url: Optional[str] = None
    
    def _fetch_directory(self) -> Dict:
        """Fetch ACME directory endpoints."""
        if self._directory:
            return self._directory
        
        try:
            response = requests.get(self.directory_url, verify=self.verify_ssl, timeout=30)
            response.raise_for_status()
            self._directory = response.json()
            return self._directory
        except Exception as e:
            logger.error(f"Failed to fetch ACME directory: {e}")
            raise
    
    def _get_nonce(self) -> str:
        """Get a fresh nonce from the ACME server."""
        if self._nonce:
            nonce = self._nonce
            self._nonce = None
            return nonce
        
        directory = self._fetch_directory()
        response = requests.head(
            directory["newNonce"],
            verify=self.verify_ssl,
            timeout=30
        )
        return response.headers["Replay-Nonce"]
    
    def _save_nonce(self, response: requests.Response) -> None:
        """Save nonce from response for reuse."""
        if "Replay-Nonce" in response.headers:
            self._nonce = response.headers["Replay-Nonce"]
    
    def _load_or_create_account_key(self) -> Any:
        """Load existing or create new ACME account key."""
        if not HAS_JOSE:
            raise RuntimeError("josepy required for ACME account operations")
        
        if self._account_key:
            return self._account_key
        
        if self.account_key_path and self.account_key_path.exists():
            # Load existing key
            key_data = self.account_key_path.read_bytes()
            private_key = serialization.load_pem_private_key(
                key_data, password=None, backend=default_backend()
            )
            self._account_key = jose.JWKRSA(key=private_key)
        else:
            # Generate new key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self._account_key = jose.JWKRSA(key=private_key)
            
            # Save key
            if self.account_key_path:
                self.account_key_path.parent.mkdir(parents=True, exist_ok=True)
                key_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                self.account_key_path.write_bytes(key_pem)
                logger.info(f"Saved ACME account key to {self.account_key_path}")
        
        return self._account_key
    
    def _signed_request(
        self,
        url: str,
        payload: Optional[Dict] = None,
        kid: Optional[str] = None,
    ) -> requests.Response:
        """Make a signed JWS request to ACME server."""
        if not HAS_JOSE:
            raise RuntimeError("josepy required for signed requests")
        
        account_key = self._load_or_create_account_key()
        nonce = self._get_nonce()
        
        # Build protected header
        header = {
            "alg": "RS256",
            "nonce": nonce,
            "url": url,
        }
        
        if kid:
            header["kid"] = kid
        else:
            header["jwk"] = account_key.public_key().fields_to_partial_json()
        
        # Encode payload
        if payload is None:
            payload_b64 = ""
        else:
            payload_json = json.dumps(payload).encode()
            payload_b64 = base64.urlsafe_b64encode(payload_json).rstrip(b"=").decode()
        
        protected_b64 = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).rstrip(b"=").decode()
        
        # Sign
        signing_input = f"{protected_b64}.{payload_b64}".encode()
        signature = account_key.sign(signing_input)
        signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
        
        # Make request
        jws = {
            "protected": protected_b64,
            "payload": payload_b64,
            "signature": signature_b64,
        }
        
        response = requests.post(
            url,
            json=jws,
            headers={"Content-Type": "application/jose+json"},
            verify=self.verify_ssl,
            timeout=30,
        )
        
        self._save_nonce(response)
        return response
    
    def register_account(self, email: str) -> str:
        """
        Register or retrieve ACME account.
        
        Returns account URL.
        """
        directory = self._fetch_directory()
        
        payload = {
            "termsOfServiceAgreed": True,
            "contact": [f"mailto:{email}"],
        }
        
        response = self._signed_request(directory["newAccount"], payload)
        
        if response.status_code in [200, 201]:
            self._account_url = response.headers.get("Location")
            logger.info(f"ACME account registered: {self._account_url}")
            return self._account_url
        else:
            raise RuntimeError(f"Account registration failed: {response.text}")
    
    def create_order(self, domains: List[str]) -> ACMEOrder:
        """
        Create a new certificate order.
        
        Args:
            domains: List of domain names (first is primary)
            
        Returns:
            ACMEOrder with authorization URLs
        """
        if not self._account_url:
            raise RuntimeError("Account not registered. Call register_account first.")
        
        directory = self._fetch_directory()
        
        identifiers = [{"type": "dns", "value": d} for d in domains]
        payload = {"identifiers": identifiers}
        
        response = self._signed_request(
            directory["newOrder"],
            payload,
            kid=self._account_url
        )
        
        if response.status_code != 201:
            raise RuntimeError(f"Order creation failed: {response.text}")
        
        order_data = response.json()
        order_url = response.headers.get("Location")
        
        return ACMEOrder(
            order_url=order_url,
            status=order_data["status"],
            identifiers=order_data["identifiers"],
            authorizations=order_data["authorizations"],
            finalize_url=order_data["finalize"],
            certificate_url=order_data.get("certificate"),
        )
    
    def get_challenges(self, order: ACMEOrder) -> List[ACMEChallenge]:
        """
        Get challenges for all authorizations in an order.
        
        Returns list of challenges (http-01 preferred).
        """
        challenges = []
        
        for auth_url in order.authorizations:
            response = self._signed_request(auth_url, None, kid=self._account_url)
            
            if response.status_code != 200:
                continue
            
            auth_data = response.json()
            domain = auth_data["identifier"]["value"]
            
            for challenge in auth_data["challenges"]:
                if challenge["type"] in ["http-01", "dns-01"]:
                    key_auth = self._compute_key_authorization(challenge["token"])
                    
                    challenges.append(ACMEChallenge(
                        type=challenge["type"],
                        url=challenge["url"],
                        token=challenge["token"],
                        status=challenge["status"],
                        key_authorization=key_auth,
                        domain=domain,
                    ))
        
        return challenges
    
    def _compute_key_authorization(self, token: str) -> str:
        """Compute key authorization for a challenge token."""
        if not HAS_JOSE:
            return token
        
        account_key = self._load_or_create_account_key()
        thumbprint = account_key.thumbprint()
        thumbprint_b64 = base64.urlsafe_b64encode(thumbprint).rstrip(b"=").decode()
        
        return f"{token}.{thumbprint_b64}"
    
    def respond_challenge(self, challenge: ACMEChallenge) -> bool:
        """
        Notify ACME server that challenge is ready.
        
        The challenge response must already be in place before calling.
        """
        response = self._signed_request(challenge.url, {}, kid=self._account_url)
        
        if response.status_code == 200:
            return True
        
        logger.error(f"Challenge response failed: {response.text}")
        return False
    
    def poll_order(self, order: ACMEOrder, timeout: int = 120) -> ACMEOrder:
        """
        Poll order status until ready or timeout with exponential backoff.
        """
        start = time.time()
        poll_interval = 1  # Start with 1 second
        max_poll_interval = 10  # Cap at 10 seconds

        while time.time() - start < timeout:
            response = self._signed_request(order.order_url, None, kid=self._account_url)

            if response.status_code != 200:
                raise RuntimeError(f"Order poll failed: {response.text}")

            order_data = response.json()
            order.status = order_data["status"]
            order.certificate_url = order_data.get("certificate")

            if order.status == "ready":
                return order
            elif order.status == "valid":
                return order
            elif order.status == "invalid":
                raise RuntimeError("Order became invalid")

            # Exponential backoff with cap
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 2, max_poll_interval)

        raise RuntimeError("Order poll timeout")
    
    def finalize_order(self, order: ACMEOrder, csr_pem: str) -> ACMEOrder:
        """
        Finalize order by submitting CSR.
        
        Args:
            order: Order in "ready" status
            csr_pem: PEM-encoded CSR
            
        Returns:
            Updated order with certificate URL
        """
        # Convert PEM to DER and base64url encode
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        
        csr = x509.load_pem_x509_csr(csr_pem.encode(), default_backend())
        csr_der = csr.public_bytes(serialization.Encoding.DER)
        csr_b64 = base64.urlsafe_b64encode(csr_der).rstrip(b"=").decode()
        
        payload = {"csr": csr_b64}
        
        response = self._signed_request(order.finalize_url, payload, kid=self._account_url)
        
        if response.status_code != 200:
            raise RuntimeError(f"Order finalization failed: {response.text}")
        
        order_data = response.json()
        order.status = order_data["status"]
        order.certificate_url = order_data.get("certificate")
        
        return order
    
    def download_certificate(self, order: ACMEOrder) -> str:
        """
        Download issued certificate.
        
        Returns PEM-encoded certificate chain.
        """
        if not order.certificate_url:
            raise RuntimeError("No certificate URL in order")
        
        response = self._signed_request(order.certificate_url, None, kid=self._account_url)
        
        if response.status_code != 200:
            raise RuntimeError(f"Certificate download failed: {response.text}")
        
        return response.text

    def check_order_status(self, order_url: str) -> Dict[str, Any]:
        """Check status of an existing order."""
        if not self._account_url:
            raise RuntimeError("Account not registered")
            
        response = self._signed_request(order_url, None, kid=self._account_url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to check order status: {response.text}")
            
        return response.json()
    
    # === High-Level API ===
    
    def issue_certificate(
        self,
        domains: List[str],
        csr_pem: str,
        email: str,
        challenge_handler: callable,
        timeout: int = 180,
    ) -> ACMEResult:
        """
        Complete certificate issuance flow.
        
        Args:
            domains: List of domain names
            csr_pem: PEM-encoded CSR
            email: Contact email for account
            challenge_handler: Callable to set up challenges
            timeout: Total timeout in seconds
            
        Returns:
            ACMEResult with certificate
        """
        try:
            # Register account
            self.register_account(email)
            
            # Create order
            order = self.create_order(domains)
            logger.info(f"Created order: {order.order_url}")
            
            # Get and respond to challenges
            challenges = self.get_challenges(order)
            
            for challenge in challenges:
                if challenge.status == "valid":
                    continue
                
                # Let handler set up the challenge response
                challenge_handler(challenge)
                
                # Notify server
                self.respond_challenge(challenge)
            
            # Wait for order to be ready
            order = self.poll_order(order, timeout=timeout)
            
            # Finalize with CSR
            if order.status == "ready":
                order = self.finalize_order(order, csr_pem)
            
            # Wait for certificate
            order = self.poll_order(order, timeout=30)
            
            if order.status != "valid":
                return ACMEResult(success=False, message=f"Order status: {order.status}")
            
            # Download certificate
            cert_pem = self.download_certificate(order)
            
            return ACMEResult(
                success=True,
                message="Certificate issued successfully",
                certificate_pem=cert_pem,
            )
            
        except Exception as e:
            logger.error(f"ACME issuance failed: {e}")
            return ACMEResult(success=False, message=str(e))
