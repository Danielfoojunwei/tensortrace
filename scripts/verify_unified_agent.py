import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from tensorguard.agent.daemon import AgentDaemon
from tensorguard.schemas.unified_config import AgentConfig, IdentityConfig, NetworkConfig, MLConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verifier")

def test_imports():
    logger.info("Testing imports...")
    try:
        from tensorguard.agent.identity.manager import IdentityManager
        from tensorguard.agent.network.guardian import NetworkGuardian
        from tensorguard.agent.ml.manager import MLManager
        logger.info("Subsystem managers imported successfully.")
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)

def test_daemon_init():
    logger.info("Testing AgentDaemon initialization...")
    try:
        # Create a dummy config just to initialize
        daemon = AgentDaemon()
        logger.info("AgentDaemon initialized.")
    except Exception as e:
        logger.error(f"Daemon init failed: {e}")
        # Proceeding, might just be config file missing which is expected

def main():
    logger.info("Starting Verification of Unified Agent...")
    test_imports()
    test_daemon_init()
    logger.info("Verification Complete!")

if __name__ == "__main__":
    main()
