
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time
from ..evidence.canonical import canonical_bytes
from ..evidence.store import get_store
import secrets

class PackageManifest(BaseModel):
    tgsp_version: str = "0.2"
    package_id: str = Field(default_factory=lambda: secrets.token_hex(8))
    model_name: str = "unknown"
    model_version: str = "0.0.1"
    author_id: str = "anonymous"
    producer_pubkey_ed25519: Optional[str] = None # Base64 encoded
    created_at: float = Field(default_factory=time.time)
    
    payload_hash: str = "pending" # SHA-256 of encrypted payload (or compressed if v0.1)
    
    content_index: List[Dict[str, str]] = [] # [{path, sha256}]
    
    policy_constraints: Dict[str, Any] = {}
    build_info: Dict[str, str] = {}
    compat_base_model_id: List[str] = [] # For backward compatibility
    
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.model_dump())
        
    def to_canonical_cbor(self) -> bytes:
        return self.canonical_bytes()
        
    def get_hash(self) -> str:
        import hashlib
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
