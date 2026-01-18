"""
Inference Router
Decides whether to route requests to FHE (Moai) or other runtimes (TEE, Plaintext).
"""

from typing import Literal

class InferenceRouter:
    """
    Policy engine for routing inference requests.
    """
    
    def __init__(self):
        pass
        
    def route(self, request_metadata: dict) -> Literal["MOAI_FHE", "SGX_TEE", "LOCAL_PLAINTEXT"]:
        """
        Determine execution environment based on privacy SLA.
        """
        privacy_level = request_metadata.get("privacy_level", "standard")
        
        if privacy_level == "critical":
            return "MOAI_FHE"
        elif privacy_level == "confidential":
            return "SGX_TEE"
        else:
            return "LOCAL_PLAINTEXT"
