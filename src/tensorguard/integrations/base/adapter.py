from abc import ABC, abstractmethod
from typing import Any, Dict

class IntegrationAdapter(ABC):
    """
    Base interface for external platform connectors.
    """
    @abstractmethod
    def connect(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    def push_artifact(self, artifact_path: str, metadata: Dict[str, Any]):
        pass

    @abstractmethod
    def pull_artifact(self, artifact_id: str) -> str:
        pass

class HelloWorldAdapter(IntegrationAdapter):
    def connect(self, config: Dict[str, Any]):
        print(f"HelloWorldAdapter connected with config: {config}")

    def push_artifact(self, artifact_path: str, metadata: Dict[str, Any]):
        print(f"HelloWorldAdapter pushing {artifact_path}")
        return "hw-artifact-123"

    def pull_artifact(self, artifact_id: str) -> str:
        print(f"HelloWorldAdapter pulling {artifact_id}")
        return "/tmp/hw-download"
