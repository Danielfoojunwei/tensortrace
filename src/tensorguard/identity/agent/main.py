"""
Machine Identity Guard - Agent Entrypoint

Allows running the identity agent as a standalone module.
Supports configuration via CLI arguments or JSON config file.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from ...agent.daemon import AgentDaemon
from ...agent.config_manager import ConfigManager
from ...schemas.unified_config import AgentConfig

def parse_args():
    parser = argparse.ArgumentParser(description="TensorGuard Identity Agent")
    parser.add_argument("--config", type=str, help="Path to agent_config.json")
    parser.add_argument("--fleet-id", type=str, help="Override fleet ID")
    parser.add_argument("--api-key", type=str, help="Override fleet API key")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("tensorguard.identity.agent")
    
    logger.info("Initializing TensorGuard Identity Agent...")
    
    # Load config
    config_path = args.config or os.path.join("configs", "agent_config.json")
    config_manager = ConfigManager(config_path=config_path, fleet_api_key=args.api_key)
    
    agent_config = config_manager.load_local()
    
    # CLI Overrides
    if args.fleet_id:
        agent_config.fleet_id = args.fleet_id
    if args.api_key:
        agent_config.api_key = args.api_key
        config_manager.fleet_api_key = args.api_key
        
    # Ensure identity is enabled if running this specific main
    agent_config.identity.enabled = True
    
    # Create a minimal daemon if we only want identity, 
    # but the current architecture uses AgentDaemon to manage subsystems.
    # However, Phase 4 says "Starts IdentityManager (scanner + heartbeat) & Starts WorkPoller".
    # For now, we reuse AgentDaemon but we can specialize it later if needed.
    
    daemon = AgentDaemon(config_path=config_path)
    # Patch the daemon's config manager and config with our overrides
    daemon.config_manager = config_manager
    # We need to ensure _init_subsystems uses our agent_config
    
    logger.info(f"Agent {agent_config.agent_name} starting for fleet {agent_config.fleet_id}")
    
    # Start the daemon
    try:
        daemon.start()
    except KeyboardInterrupt:
        daemon.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()
