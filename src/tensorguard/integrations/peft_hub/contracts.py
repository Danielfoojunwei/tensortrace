from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ConnectorValidationResult(BaseModel):
    ok: bool
    details: str
    remediation: Optional[str] = None

class Connector(ABC):
    """Base interface for all PEFT connectors."""
    
    @property
    @abstractmethod
    def id(self) -> str: pass
    
    @property
    @abstractmethod
    def name(self) -> str: pass
    
    @property
    @abstractmethod
    def category(self) -> str: pass # training, data, tracking, etc.

    @abstractmethod
    def check_installed(self) -> bool:
        """Check if any required binary/library dependencies are present."""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> ConnectorValidationResult:
        """Validate the specific configuration provided by the user."""
        pass

    @abstractmethod
    def to_runtime(self, config: Dict[str, Any]) -> Any:
        """Convert config to a runtime object or command wrapper."""
        pass
