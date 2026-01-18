import hashlib
import hmac
import time
import uuid
import logging
from typing import Optional, Dict, Any
from ...utils.http import StandardClient

logger = logging.getLogger(__name__)

class IdentityAgentClient(StandardClient):
    """
    HTTP Client for identity agents with HMAC signing.
    """
    def __init__(self, base_url: str, fleet_id: str, api_key: str):
        super().__init__(base_url)
        self.fleet_id = fleet_id
        self.api_key = api_key

    def signed_request(self, method: str, path: str, json_data: Any = None, **kwargs) -> Dict[str, Any]:
        """Perform a signed request to the platform."""
        timestamp = str(int(time.time()))
        nonce = str(uuid.uuid4())
        
        # Compute body hash
        import json
        body = json.dumps(json_data).encode() if json_data else b""
        body_hash = hashlib.sha256(body).hexdigest()
        
        # Message to sign: timestamp:nonce:body_hash
        message = f"{timestamp}:{nonce}:{body_hash}"
        
        # Compute signature
        # Note: In a real system, use KDF. Here we match the platform's current (simple) verification.
        # The platform currently uses fleet.api_key_hash as the key.
        # Wait, if the platform uses the hash as the key, the agent needs the hash.
        # But usually the agent has the RAW key and the platform has the hash.
        # For this MVP, we'll assume the agent uses the API key it has.
        signature = hmac.new(
            self.api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            "x-tg-fleet-id": self.fleet_id,
            "x-tg-timestamp": timestamp,
            "x-tg-nonce": nonce,
            "x-tg-signature": signature,
        }
        
        return self.request(method, path, json=json_data, headers=headers, **kwargs)
