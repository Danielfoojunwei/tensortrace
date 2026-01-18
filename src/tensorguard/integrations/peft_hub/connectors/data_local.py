import os
import logging
from typing import Dict, Any
from ..contracts import Connector, ConnectorValidationResult
from ..catalog import ConnectorCatalog

logger = logging.getLogger(__name__)

class LocalDataConnector(Connector):
    @property
    def id(self) -> str: return "data_local"
    @property
    def name(self) -> str: return "Local Filesystem Dataset"
    @property
    def category(self) -> str: return "data"

    def check_installed(self) -> bool: return True # Always native

    def validate_config(self, config: Dict[str, Any]) -> ConnectorValidationResult:
        path = config.get("dataset_name_or_path")
        if not path:
            return ConnectorValidationResult(ok=False, details="Missing path")
        return ConnectorValidationResult(ok=True, details="Valid")

    def to_runtime(self, config: Dict[str, Any]) -> str:
        return config.get("dataset_name_or_path")

class LocalStoreConnector(Connector):
    @property
    def id(self) -> str: return "store_local"
    @property
    def name(self) -> str: return "Local Artifact Store"
    @property
    def category(self) -> str: return "store"

    def check_installed(self) -> bool: return True

    def validate_config(self, config: Dict[str, Any]) -> ConnectorValidationResult:
        path = config.get("path", "./runs")
        return ConnectorValidationResult(ok=True, details="Valid")

    def to_runtime(self, config: Dict[str, Any]) -> str:
        return config.get("path", "./runs")

# Auto-register
ConnectorCatalog.register(LocalDataConnector())
ConnectorCatalog.register(LocalStoreConnector())
