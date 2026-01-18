"""
Common Schemas

Shared data models used across Agent and Core.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

import uuid

class Demonstration(BaseModel):
    """A collected demonstration (visual/action data)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: Optional[str] = None
    instruction: Optional[str] = None
    observations: Optional[List[Any]] = None
    actions: Optional[List[Any]] = None
    data: Any = None
    metadata: Dict[str, Any] = {}

class ShieldConfig(BaseModel):
    """Configuration for Privacy Shield (Legacy)."""
    model_type: str = "pi0"
    key_path: str = "keys/tensorguard.key"
    security_level: int = 2
    max_gradient_norm: float = 1.0
    dp_epsilon: float = 10.0
    sparsity: float = 0.5
    compression_ratio: float = 4.0

class ClientStatus(BaseModel):
    """Status report from edge client."""
    pending_submissions: int
    total_submissions: int
    privacy_budget_remaining: float
    last_model_version: str
    connection_status: str

class SubmissionReceipt(BaseModel):
    """Receipt for a submitted update."""
    submission_id: str
    status: str
    timestamp: float
