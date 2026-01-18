"""
Unified Configuration Schemas

Defines the configuration structure for the TensorGuard Unified Agent and Control Plane.
Consolidates settings for Identity (CLM), Network (RTPL), and Machine Learning (FL/PEFT).
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator


# === Identity (CLM) Configuration ===

class KeyType(str, Enum):
    RSA = "RSA"
    ECDSA = "ECDSA"

class IdentityConfig(BaseModel):
    """Configuration for Machine Identity Guard."""
    enabled: bool = True
    
    # Key Management
    key_storage_path: str = "/var/lib/tensorguard/keys"
    key_type: KeyType = KeyType.RSA
    key_size: int = 2048
    
    # Scanner
    scan_interval_seconds: int = 3600  # 1 hour
    scan_kubernetes: bool = True
    scan_nginx: bool = True
    scan_envoy: bool = True
    scan_filesystem: bool = False
    
    # Automation
    auto_renew: bool = True
    auto_deploy: bool = True


# === Network (RTPL) Configuration ===

class DefenseMode(str, Enum):
    NONE = "none"
    PADDING = "padding"
    WTF_PAD = "wtf_pad"
    FRONT = "front"

class NetworkConfig(BaseModel):
    """Configuration for Robot Traffic Privacy Layer."""
    enabled: bool = False
    
    # Proxy Settings
    proxy_port: int = 9000
    target_host: str = "localhost"
    target_port: int = 8080
    
    # Defense Settings
    defense_mode: DefenseMode = DefenseMode.FRONT
    
    # Padding Config
    padding_bucket_size: int = 800
    
    # FRONT Config
    front_max_dummies: int = 1000
    front_min_dummies: int = 50
    
    # WTF-PAD Config
    wtf_pad_burst_threshold: float = 0.5


# === Deployment Directive (Control Plane → Agent) ===

class DeploymentDirective(BaseModel):
    """
    Deployment directive from control plane to agent.

    Specifies which model/adapter version the agent should apply.
    Supports canary, A/B, shadow, and full deployment modes.
    """
    deployment_id: str
    target_adapter_id: Optional[str] = None
    target_model_version: str
    shadow: bool = False  # If true, run both adapters, discard new output (shadow mode)
    compat_min_version: str = "1.0.0"  # Minimum agent version allowed
    rollback_adapter_id: Optional[str] = None  # Previous adapter to rollback to


# === Machine Learning (Core) Configuration ===

class ModelType(str, Enum):
    PI0 = "pi0"
    OPEN_VLA = "open_vla"
    OCTO = "octo"

class SecurityLevel(str, Enum):
    HIGH = "high"      # Full FHE + DP
    MEDIUM = "medium"  # DP + TLS
    LOW = "none"       # Standard

class MLConfig(BaseModel):
    """Configuration for Federated Learning & PEFT."""
    enabled: bool = True
    
    # Model Settings
    model_type: ModelType = ModelType.PI0
    
    # Privacy Settings
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    dp_epsilon: float = 3.0
    max_gradient_norm: float = 1.0
    
    # Optimization
    compression_ratio: float = 0.1  # Top-k encoded
    sparsity: float = 0.5
    
    # Resource Limits
    max_cpu_percent: float = 80.0
    max_memory_gb: int = 4
    
    # Aggregation
    aggregator_url: str = "127.0.0.1:8080"


# === Agent Configuration (Client-Side) ===

class AgentConfig(BaseModel):
    """
    Master configuration for the Unified Agent.
    
    This config is applied to the agent daemon.
    """
    # Identity
    agent_id: Optional[str] = None
    agent_name: str
    fleet_id: str
    tenant_id: Optional[str] = None
    
    # Connectivity
    control_plane_url: str = "https://api.tensorguard.io"
    api_key: Optional[str] = None  # Fleet API Key
    
    # Modules
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    # Deployment Directive (from control plane)
    deployment: Optional[DeploymentDirective] = None

    # System
    log_level: str = "INFO"
    data_dir: str = "/var/lib/tensorguard"


# === Fleet Policy (Server-Side) ===

class FleetPolicy(BaseModel):
    """
    Policy defined on the Control Plane and pushed to Agents.
    
    Overrides agent local defaults.
    """
    fleet_id: str
    name: str
    
    # Global overrides
    force_defense_mode: Optional[DefenseMode] = None
    min_dp_epsilon: Optional[float] = None
    required_scanners: List[str] = []
    
    # Module policies
    identity_rules: Dict[str, Any] = {}
    network_rules: Dict[str, Any] = {}
    ml_rules: Dict[str, Any] = {}
