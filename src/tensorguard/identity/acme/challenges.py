"""
ACME Challenge Handlers - HTTP-01 and DNS-01 Challenge Responders

Provides mechanisms to respond to ACME challenges for domain validation.
"""

import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass
import logging

from ...utils.production_gates import ProductionGateError, is_production, require_dependency

logger = logging.getLogger(__name__)


@dataclass
class ChallengeResponse:
    """Challenge response data."""
    domain: str
    token: str
    key_authorization: str
    challenge_type: str


class HTTP01Handler(BaseHTTPRequestHandler):
    """
    HTTP request handler for ACME HTTP-01 challenges.
    
    Serves key authorization at /.well-known/acme-challenge/{token}
    """
    
    # Class-level challenge storage
    challenges: Dict[str, str] = {}
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path.startswith("/.well-known/acme-challenge/"):
            token = self.path.split("/")[-1]
            
            if token in self.challenges:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(self.challenges[token].encode())
                logger.info(f"Served challenge for token: {token[:20]}...")
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class HTTP01Server:
    """
    HTTP-01 challenge server.
    
    Runs a simple HTTP server to respond to ACME HTTP-01 challenges.
    Typically runs on port 80 or behind a reverse proxy.
    """
    
    def __init__(self, port: int = 80, bind: str = "0.0.0.0"):
        self.port = port
        self.bind = bind
        self.server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
    
    def add_challenge(self, token: str, key_authorization: str) -> None:
        """Register a challenge response."""
        HTTP01Handler.challenges[token] = key_authorization
        logger.info(f"Added HTTP-01 challenge: {token[:20]}...")
    
    def remove_challenge(self, token: str) -> None:
        """Remove a challenge response."""
        if token in HTTP01Handler.challenges:
            del HTTP01Handler.challenges[token]
    
    def start(self) -> None:
        """Start the challenge server in a background thread."""
        self.server = HTTPServer((self.bind, self.port), HTTP01Handler)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"HTTP-01 challenge server started on {self.bind}:{self.port}")
    
    def stop(self) -> None:
        """Stop the challenge server."""
        if self.server:
            self.server.shutdown()
            self.server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("HTTP-01 challenge server stopped")


class FileBasedHTTP01:
    """
    File-based HTTP-01 challenge responder.
    
    Writes challenge files to a webroot directory for serving
    by an existing web server (Nginx, Apache, etc.).
    """
    
    def __init__(self, webroot: str = "/var/www/html"):
        self.webroot = Path(webroot)
        self.challenge_dir = self.webroot / ".well-known" / "acme-challenge"
    
    def add_challenge(self, token: str, key_authorization: str) -> Path:
        """Write challenge file to webroot."""
        self.challenge_dir.mkdir(parents=True, exist_ok=True)
        
        challenge_file = self.challenge_dir / token
        challenge_file.write_text(key_authorization)
        os.chmod(challenge_file, 0o644)
        
        logger.info(f"Wrote HTTP-01 challenge file: {challenge_file}")
        return challenge_file
    
    def remove_challenge(self, token: str) -> None:
        """Remove challenge file."""
        challenge_file = self.challenge_dir / token
        if challenge_file.exists():
            challenge_file.unlink()
            logger.info(f"Removed HTTP-01 challenge file: {challenge_file}")


class DNS01Handler:
    """
    DNS-01 challenge handler interface.
    
    DNS-01 challenges require setting a TXT record at:
    _acme-challenge.{domain}
    
    This is an abstract interface - implementations provided for
    specific DNS providers.
    """
    
    def compute_txt_value(self, key_authorization: str) -> str:
        """Compute the TXT record value from key authorization."""
        import hashlib
        import base64
        
        digest = hashlib.sha256(key_authorization.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    
    def add_txt_record(self, domain: str, txt_value: str) -> bool:
        """Add TXT record. Override in provider implementations."""
        raise ProductionGateError(
            gate_name="DNS_PROVIDER",
            message="DNS provider not configured for DNS-01 challenges.",
            remediation="Configure a supported DNS provider (e.g., Route53 or Cloudflare) or use HTTP-01 challenges.",
        )
    
    def remove_txt_record(self, domain: str) -> bool:
        """Remove TXT record. Override in provider implementations."""
        raise ProductionGateError(
            gate_name="DNS_PROVIDER",
            message="DNS provider not configured for DNS-01 challenges.",
            remediation="Configure a supported DNS provider (e.g., Route53 or Cloudflare) or use HTTP-01 challenges.",
        )


class ManualDNS01Handler(DNS01Handler):
    """
    Manual DNS-01 handler - prompts user to set TXT record.
    
    Useful for initial setup or when automatic DNS is not available.
    """
    
    def __init__(self, callback: Optional[Callable[[str, str], bool]] = None):
        self.callback = callback
    
    def add_txt_record(self, domain: str, txt_value: str) -> bool:
        """Notify user to add TXT record."""
        record_name = f"_acme-challenge.{domain}"
        
        if self.callback:
            return self.callback(record_name, txt_value)
        
        logger.info(f"Add DNS TXT record:\n  Name: {record_name}\n  Value: {txt_value}")
        return True
    
    def remove_txt_record(self, domain: str) -> bool:
        """Notify user to remove TXT record."""
        record_name = f"_acme-challenge.{domain}"
        logger.info(f"Remove DNS TXT record: {record_name}")
        return True


class Route53DNS01Handler(DNS01Handler):
    """
    AWS Route53 DNS-01 handler.
    
    Requires boto3 and AWS credentials.
    """
    
    def __init__(self, hosted_zone_id: str):
        self.hosted_zone_id = hosted_zone_id
    
    def add_txt_record(self, domain: str, txt_value: str) -> bool:
        """Add TXT record via Route53 API."""
        boto3 = require_dependency(
            "boto3",
            package_name="boto3",
            remediation="Install boto3 and provide AWS credentials.",
        )
        if boto3 is None:
            return False

        try:
            client = boto3.client("route53")
            record_name = f"_acme-challenge.{domain}"

            client.change_resource_record_sets(
                HostedZoneId=self.hosted_zone_id,
                ChangeBatch={
                    "Changes": [{
                        "Action": "UPSERT",
                        "ResourceRecordSet": {
                            "Name": record_name,
                            "Type": "TXT",
                            "TTL": 60,
                            "ResourceRecords": [{"Value": f'"{txt_value}"'}],
                        }
                    }]
                }
            )

            logger.info(f"Added Route53 TXT record: {record_name}")
            return True
        except Exception as e:
            logger.error(f"Route53 error: {e}")
            return False
    
    def remove_txt_record(self, domain: str) -> bool:
        """Remove TXT record via Route53 API."""
        boto3 = require_dependency(
            "boto3",
            package_name="boto3",
            remediation="Install boto3 and provide AWS credentials.",
        )
        if boto3 is None:
            return False

        try:
            client = boto3.client("route53")
            record_name = f"_acme-challenge.{domain}"

            response = client.list_resource_record_sets(
                HostedZoneId=self.hosted_zone_id,
                StartRecordName=record_name,
                StartRecordType="TXT",
            )
            records = [
                record for record in response.get("ResourceRecordSets", [])
                if record.get("Name", "").rstrip(".") == record_name and record.get("Type") == "TXT"
            ]
            if not records:
                logger.warning(f"No Route53 TXT records found for {record_name}")
                return False

            changes = [{"Action": "DELETE", "ResourceRecordSet": record} for record in records]
            client.change_resource_record_sets(
                HostedZoneId=self.hosted_zone_id,
                ChangeBatch={"Changes": changes},
            )

            logger.info(f"Removed Route53 TXT record: {record_name}")
            return True
        except Exception as e:
            logger.error(f"Route53 error: {e}")
            return False


class CloudflareDNS01Handler(DNS01Handler):
    """
    Cloudflare DNS-01 handler.
    
    Requires Cloudflare API token.
    """
    
    def __init__(self, api_token: str, zone_id: Optional[str] = None):
        self.api_token = api_token
        self.zone_id = zone_id
    
    def add_txt_record(self, domain: str, txt_value: str) -> bool:
        """Add TXT record via Cloudflare API."""
        requests = require_dependency(
            "requests",
            package_name="requests",
            remediation="Install requests: pip install requests",
        )
        if requests is None:
            return False

        try:
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }

            record_name = f"_acme-challenge.{domain}"

            if not self.zone_id:
                raise ProductionGateError(
                    gate_name="CLOUDFLARE_ZONE",
                    message="Cloudflare zone_id is required for DNS-01 challenges.",
                    remediation="Provide zone_id for CloudflareDNS01Handler.",
                )

            url = f"https://api.cloudflare.com/client/v4/zones/{self.zone_id}/dns_records"

            response = requests.post(
                url,
                headers=headers,
                json={
                    "type": "TXT",
                    "name": record_name,
                    "content": txt_value,
                    "ttl": 60,
                },
                timeout=30,
            )

            if response.ok:
                logger.info(f"Added Cloudflare TXT record: {record_name}")
                return True
            logger.error(f"Cloudflare error: {response.text}")
            return False

        except Exception as e:
            logger.error(f"Cloudflare error: {e}")
            return False
    
    def remove_txt_record(self, domain: str) -> bool:
        """Remove TXT record via Cloudflare API."""
        requests = require_dependency(
            "requests",
            package_name="requests",
            remediation="Install requests: pip install requests",
        )
        if requests is None:
            return False

        try:
            if not self.zone_id:
                raise ProductionGateError(
                    gate_name="CLOUDFLARE_ZONE",
                    message="Cloudflare zone_id is required for DNS-01 challenges.",
                    remediation="Provide zone_id for CloudflareDNS01Handler.",
                )

            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }
            record_name = f"_acme-challenge.{domain}"
            list_url = f"https://api.cloudflare.com/client/v4/zones/{self.zone_id}/dns_records"
            response = requests.get(
                list_url,
                headers=headers,
                params={"type": "TXT", "name": record_name},
                timeout=30,
            )
            response.raise_for_status()
            results = response.json().get("result", [])
            if not results:
                logger.warning(f"No Cloudflare TXT records found for {record_name}")
                return False

            for record in results:
                record_id = record["id"]
                delete_url = f"https://api.cloudflare.com/client/v4/zones/{self.zone_id}/dns_records/{record_id}"
                delete_response = requests.delete(delete_url, headers=headers, timeout=30)
                if not delete_response.ok:
                    logger.error(f"Cloudflare delete failed: {delete_response.text}")
                    return False

            logger.info(f"Removed Cloudflare TXT record: {record_name}")
            return True
        except Exception as e:
            logger.error(f"Cloudflare error: {e}")
            return False


class ChallengeCoordinator:
    """
    Coordinates challenge setup and teardown.
    
    Works with the ACME client to handle challenges automatically.
    """
    
    def __init__(
        self,
        http01_handler: Optional[Any] = None,
        dns01_handler: Optional[DNS01Handler] = None,
    ):
        self.http01_handler = http01_handler
        self.dns01_handler = dns01_handler
        self._active_challenges: Dict[str, Dict] = {}
    
    def setup_challenge(self, challenge: Any) -> bool:
        """
        Set up a challenge response.
        
        Args:
            challenge: ACMEChallenge object
            
        Returns:
            True if setup successful
        """
        if challenge.type == "http-01":
            if self.http01_handler:
                self.http01_handler.add_challenge(
                    challenge.token,
                    challenge.key_authorization
                )
                self._active_challenges[challenge.token] = {
                    "type": "http-01",
                    "token": challenge.token,
                }
                return True
        
        elif challenge.type == "dns-01":
            if self.dns01_handler:
                txt_value = self.dns01_handler.compute_txt_value(
                    challenge.key_authorization
                )
                result = self.dns01_handler.add_txt_record(
                    challenge.domain,
                    txt_value
                )
                if result:
                    self._active_challenges[challenge.token] = {
                        "type": "dns-01",
                        "domain": challenge.domain,
                    }
                return result
        
        return False
    
    def cleanup_challenge(self, challenge: Any) -> None:
        """Clean up a challenge response."""
        if challenge.token in self._active_challenges:
            info = self._active_challenges[challenge.token]
            
            if info["type"] == "http-01" and self.http01_handler:
                self.http01_handler.remove_challenge(info["token"])
            elif info["type"] == "dns-01" and self.dns01_handler:
                self.dns01_handler.remove_txt_record(info["domain"])
            
            del self._active_challenges[challenge.token]
    
    def cleanup_all(self) -> None:
        """Clean up all active challenges."""
        for token, info in list(self._active_challenges.items()):
            if info["type"] == "http-01" and self.http01_handler:
                self.http01_handler.remove_challenge(token)
            elif info["type"] == "dns-01" and self.dns01_handler:
                self.dns01_handler.remove_txt_record(info["domain"])
        
        self._active_challenges.clear()
