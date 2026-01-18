"""
Unified Key Management Fabric (UKF)
Consolidates storage, metadata, and lifecycle for all TensorGuard keys.
"""

import os
import json
import datetime
import secrets
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union

from ..utils.logging import get_logger
from ..utils.exceptions import CryptographyError

logger = get_logger(__name__)

class KeyScope(str, Enum):
    IDENTITY = "identity"      # Ed25519, X25519, RSA/EC Certificates
    INFERENCE = "inference"    # MOAI (CKKS) contexts
    AGGREGATION = "aggregation" # N2HE (LWE) keys
    SYSTEM = "system"          # Internal app secrets

@dataclass
class UnifiedKeyMetadata:
    """Standard metadata for any key in the TensorGuard fabric."""
    key_id: str
    scope: KeyScope
    algorithm: str
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    owner_id: str = "agent-local"
    version: str = "2.0.0"
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

class UnifiedKeyManager:
    """
    Central orchestrator for key persistence and discovery.
    Ensures consistent pathing, permissions, and metadata across all subsystems.
    """
    
    def __init__(self, vault_root: str = "keys"):
        self.vault_root = Path(vault_root)
        self.vault_root.mkdir(parents=True, exist_ok=True)
        # Ensure base dir permissions if possible
        try: os.chmod(self.vault_root, 0o700)
        except: pass

    def _get_scope_path(self, scope: KeyScope) -> Path:
        path = self.vault_root / scope.value
        path.mkdir(exist_ok=True)
        return path

    def save_key_artifact(self, 
                       scope: KeyScope, 
                       name: str, 
                       data: bytes, 
                       algorithm: str,
                       params: Optional[Dict[str, Any]] = None,
                       suffix: str = ".bin") -> str:
        """
        Save a binary key artifact with accompanying metadata.
        """
        name = str(name)
        scope_dir = self._get_scope_path(scope)
        
        # Path safety
        if ".." in name or name.startswith("/") or name.startswith("\\"):
            name = os.path.basename(name)
            
        key_id = f"{scope.value}_{name}"
        data_path = scope_dir / f"{name}{suffix}"
        meta_path = scope_dir / f"{name}.meta.json"
        
        # 1. Update/Create Metadata
        meta = UnifiedKeyMetadata(
            key_id=key_id,
            scope=scope,
            algorithm=algorithm,
            params=params or {}
        )
        
        try:
            # 2. Save Binary
            data_path.write_bytes(data)
            data_path.chmod(0o600)
            
            # 3. Save Metadata
            meta_path.write_text(meta.to_json())
            meta_path.chmod(0o600)
            
            logger.info(f"Saved {scope.value} key '{name}' to {data_path}")
            return key_id
        except Exception as e:
            raise CryptographyError(f"Vault failure for {name}: {e}")

    def load_key_artifact(self, scope: KeyScope, name: str, suffix: str = ".bin") -> Tuple[bytes, UnifiedKeyMetadata]:
        """Load a key and its metadata."""
        name = str(name)
        scope_dir = self._get_scope_path(scope)
        data_path = scope_dir / f"{name}{suffix}"
        meta_path = scope_dir / f"{name}.meta.json"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Key file not found: {data_path}")
            
        try:
            data = data_path.read_bytes()
            meta_data = json.loads(meta_path.read_text())
            # Convert scope string back to Enum
            meta_data['scope'] = KeyScope(meta_data['scope'])
            meta = UnifiedKeyMetadata(**meta_data)
            return data, meta
        except Exception as e:
            raise CryptographyError(f"Failed to load key {name}: {e}")

    def list_keys(self, scope: Optional[KeyScope] = None) -> List[Dict[str, Any]]:
        """List all available keys, optionally filtered by scope."""
        result = []
        scopes = [scope] if scope else list(KeyScope)
        
        for s in scopes:
            scope_dir = self.vault_root / s.value
            if not scope_dir.exists():
                continue
            
            for meta_file in scope_dir.glob("*.meta.json"):
                try:
                    meta = json.loads(meta_file.read_text())
                    result.append(meta)
                except:
                    continue
        return result

    def delete_key(self, scope: KeyScope, name: str, suffix: str = ".bin"):
        """Securely remove a key artifact."""
        name = str(name)
        scope_dir = self._get_scope_path(scope)
        (scope_dir / f"{name}{suffix}").unlink(missing_ok=True)
        (scope_dir / f"{name}.meta.json").unlink(missing_ok=True)
        logger.info(f"Deleted {scope.value} key: {name}")

# Global Instance
vault = UnifiedKeyManager()
