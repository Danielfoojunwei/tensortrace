"""
Unified Agent Daemon

The core runtime for the TensorGuard Edge Agent.
Orchestrates Identity, Network, and ML subsystems based on unified configuration.
"""

import time
import logging
import signal
import sys
import threading
import os
from typing import Optional

from .config_manager import ConfigManager
from ..schemas.unified_config import AgentConfig

# Subsystems
from .identity.manager import IdentityManager
from .network.guardian import NetworkGuardian
from .ml.manager import MLManager
from .edge_manager import EdgeAgentManager

logger = logging.getLogger(__name__)

class AgentDaemon:
    """
    Main agent process.
    """

    # Sync configuration
    SYNC_INTERVAL_SECONDS = 60
    SYNC_MAX_BACKOFF_SECONDS = 300

    def __init__(self, config_path: str = "./configs/agent_config.json"):
        self.config_manager = ConfigManager(config_path=config_path)
        self.running = False
        self._stop_event = threading.Event()

        # Threads
        self._sync_thread = None

        # Subsystems
        self.identity_mgr: Optional[IdentityManager] = None
        self.network_guard: Optional[NetworkGuardian] = None
        self.ml_mgr: Optional[MLManager] = None
        self.edge_mgr: Optional[EdgeAgentManager] = None
    
    def start(self):
        """Start the agent daemon."""
        logger.info("Starting TensorGuard Unified Agent...")
        self.running = True
        
        # 1. Load Config
        config = self.config_manager.load_local()
        logger.info(f"Loaded config for agent: {config.agent_name} ({config.fleet_id})")
        
        # 2. Initialize Subsystems
        self._init_subsystems(config)
        
        # 3. Start Subsystems
        self._start_subsystems()
        
        # 4. Register Config Listener
        self.config_manager.add_listener(self._on_config_update)
        
        # 5. Start Sync Loop
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        
        # 6. Block main thread (or wait for signal) using event-based waiting
        logger.info("Agent daemon running. Press Ctrl+C to stop.")
        try:
            # Wait on event instead of polling with sleep(1)
            self._stop_event.wait()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the agent daemon."""
        logger.info("Stopping agent...")
        self.running = False
        self._stop_event.set()  # Signal all waiting threads to stop

        # Stop subsystems
        if self.identity_mgr: self.identity_mgr.stop()
        if self.network_guard: self.network_guard.stop()
        if self.ml_mgr: self.ml_mgr.stop()
        if self.edge_mgr: self.edge_mgr.stop()
        
    def _init_subsystems(self, config: AgentConfig):
        """Initialize subsystem instances."""
        self.identity_mgr = IdentityManager(config, self.config_manager)
        self.network_guard = NetworkGuardian(config, self.config_manager)
        self.ml_mgr = MLManager(config, self.config_manager)
        self.edge_mgr = EdgeAgentManager(config, self.config_manager)
        
    def _start_subsystems(self):
        """Start enabled subsystems."""
        config = self.config_manager.current_config
        if self.identity_mgr and config.identity.enabled:
            self.identity_mgr.start()
        if self.network_guard and config.network.enabled:
            self.network_guard.start()
        if self.ml_mgr and config.ml.enabled:
            self.ml_mgr.start()
        if self.edge_mgr:
            self.edge_mgr.start()

    def _sync_loop(self):
        """Periodic sync with control plane with exponential backoff on errors."""
        consecutive_failures = 0

        while self.running:
            try:
                self.config_manager.sync_with_control_plane()
                consecutive_failures = 0  # Reset on success
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Sync loop error (attempt {consecutive_failures}): {e}")

            # Calculate sleep interval with exponential backoff on failures
            if consecutive_failures > 0:
                backoff = min(
                    self.SYNC_INTERVAL_SECONDS * (2 ** consecutive_failures),
                    self.SYNC_MAX_BACKOFF_SECONDS
                )
                sleep_interval = backoff
            else:
                sleep_interval = self.SYNC_INTERVAL_SECONDS

            # Use event-based wait for responsive shutdown
            if self._stop_event.wait(timeout=sleep_interval):
                break  # Stop event was set

    def _on_config_update(self, new_config: AgentConfig):
        """Handle hot-reload of configuration."""
        logger.info("Configuration updated, applying changes...")
        self._apply_config(new_config)

    def _apply_config(self, config: AgentConfig):
        """Apply configuration to subsystems."""
        if self.identity_mgr:
            self.identity_mgr.configure(config.identity)
            if config.identity.enabled and not self.identity_mgr.running:
                self.identity_mgr.start()
            elif not config.identity.enabled and self.identity_mgr.running:
                self.identity_mgr.stop()
            
        if self.network_guard:
            self.network_guard.configure(config.network)
            if config.network.enabled and not self.network_guard.running:
                self.network_guard.start()
            elif not config.network.enabled and self.network_guard.running:
                self.network_guard.stop()
            
        if self.ml_mgr:
            self.ml_mgr.configure(config.ml)
            if config.ml.enabled and not self.ml_mgr.running:
                self.ml_mgr.start()
            elif not config.ml.enabled and self.ml_mgr.running:
                self.ml_mgr.stop()

def main():
    """Entry point."""
    api_key = os.environ.get("TG_FLEET_API_KEY")
    if not api_key:
        # Allow running without API key for dev/test if manually config provided
        # But generally warn
        print("Warning: TG_FLEET_API_KEY environment variable not set.")
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    daemon = AgentDaemon()
    daemon.start()

if __name__ == "__main__":
    main()
