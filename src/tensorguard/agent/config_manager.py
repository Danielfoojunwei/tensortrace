"""
Configuration Manager - Agent Config Sync

Handles synchronization of configuration between the local agent and the control plane.
Persists configuration to disk for offline resilience.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from ..schemas.unified_config import AgentConfig
from ..utils.logging import get_logger
from ..utils.files import atomic_write, sanitize_path
from ..utils.http import get_standard_client

logger = get_logger(__name__)

class ConfigManager:
    """
    Manages the lifecycle of the Agent's configuration.
    """
    
    def __init__(
        self,
        config_path: str = "configs/agent_config.json",
        fleet_api_key: Optional[str] = None
    ):
        self.config_path = sanitize_path(config_path, "configs" if not os.path.isabs(config_path) else None)
        self.fleet_api_key = fleet_api_key or os.environ.get("TG_FLEET_API_KEY")
        self.current_config: Optional[AgentConfig] = None
        self._listeners: list[Callable[[AgentConfig], None]] = []
        self._http_client: Optional[Any] = None

    def _get_client(self):
        if not self._http_client and self.current_config:
            self._http_client = get_standard_client(
                self.current_config.control_plane_url, 
                self.fleet_api_key
            )
        return self._http_client

    def _save_local(self, config: AgentConfig):
        """Save configuration to disk with safety checks."""
        try:
            atomic_write(self.config_path, config.json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save local config: {e}")
        
    def add_listener(self, callback: Callable[[AgentConfig], None]):
        """Register a callback for config updates."""
        self._listeners.append(callback)
        
    def _notify_listeners(self):
        """Notify all listeners of new config."""
        if not self.current_config:
            return
        
        for listener in self._listeners:
            try:
                listener(self.current_config)
            except Exception as e:
                logger.error(f"Config listener failed: {e}")

    def load_local(self) -> AgentConfig:
        """Load configuration from local storage or defaults."""
        config_data = {}
        
        if self.config_path.exists():
            try:
                 import json
                 config_data = json.loads(self.config_path.read_text())
                 logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load local config: {e}")
        
        # Env vars override
        if not config_data.get("fleet_id"):
             config_data["fleet_id"] = os.environ.get("TG_FLEET_ID", "unknown")
            
        if not config_data.get("agent_name"):
            import socket
            config_data["agent_name"] = os.environ.get("TG_AGENT_NAME", socket.gethostname())
            
        try:
            self.current_config = AgentConfig(**config_data)
        except Exception as e:
            logger.warning(f"Invalid local config, using defaults: {e}")
            self.current_config = AgentConfig(
                agent_name="fallback",
                fleet_id="unknown"
            )
            
        return self.current_config

    def sync_with_control_plane(self) -> bool:
        """Sync configuration with the Control Plane."""
        if not self.current_config:
            self.load_local()
            
        if not self.fleet_api_key:
            logger.warning("No API key, running in offline/local mode")
            return False
            
        client = self._get_client()
        if not client:
            return False
            
        try:
            payload = {
                "name": self.current_config.agent_name,
                "fleet_id": self.current_config.fleet_id,
            }
            
            new_config_data = client.request("POST", "api/v1/config/agent/sync", json=payload)
            new_config = AgentConfig(**new_config_data)
            
            if new_config != self.current_config:
                logger.info("Configuration updated from Control Plane")
                self.current_config = new_config
                self._save_local(new_config)
                self._notify_listeners()
                # Reset client as URL might have changed
                self._http_client = None
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return False
