import importlib.util
import logging
from typing import Dict, Any, List, Type
from .contracts import Connector

logger = logging.getLogger(__name__)

class ConnectorCatalog:
    """Registry and discovery service for PEFT connectors."""
    
    _connectors: Dict[str, Connector] = {}

    @classmethod
    def register(cls, connector: Connector):
        cls._connectors[connector.id] = connector
        logger.info(f"Registered connector: {connector.id} [{connector.category}]")

    @classmethod
    def list_connectors(cls) -> List[Dict[str, Any]]:
        """List all registered connectors with their status."""
        return [
            {
                "id": c.id,
                "name": c.name,
                "category": c.category,
                "installed": c.check_installed()
            }
            for c in cls._connectors.values()
        ]

    @classmethod
    def get_connector(cls, connector_id: str) -> Connector:
        if connector_id not in cls._connectors:
            raise ValueError(f"Unknown connector: {connector_id}")
        return cls._connectors[connector_id]

# Dependency checking utility
def is_package_installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

# Placeholder to trigger registration
def discover_connectors():
    """Import all connectors to trigger registration."""
    # In a real app, we might use pkgutil to walk the 'connectors/' directory
    # For now, we manually import the core ones we're about to build
    try:
        from .connectors import training_hf, data_local
    except ImportError as e:
        logger.error(f"Failed to discover native connectors: {e}")
