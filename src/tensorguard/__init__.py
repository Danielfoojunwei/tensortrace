"""
TensorGuard: Privacy-Preserving VLA Fine-Tuning for Humanoid Robotics

Plug-and-play SDK for Robotics System Integrators to fine-tune
Vision-Language-Action models without exposing proprietary data.

Production-Grade Features:
- Operating envelope enforcement
- Canonical UpdatePackage format
- Separate privacy/training controls
- Enterprise key management
- Evaluation gating
- IL/RL pipeline separation
- SRE observability
- Resilient aggregation
"""

__version__ = "2.1.0"
__author__ = "Daniel Foo & The TensorGuard Team"

try:
    # Core interfaces
    from .core.client import EdgeClient, create_client
    from .schemas.common import ShieldConfig, Demonstration, SubmissionReceipt, ClientStatus
    from .core.adapters import VLAAdapter
    from .utils.config import settings

    # Production components (from core.production)
    from .core.production import (
        # Operating envelope
        OperatingEnvelope,
        PEFTStrategy,
        # Update package
        UpdatePackage,
        ModelTargetMap,
        TrainingMetadata,
        SafetyStatistics,
        ObjectiveType,
        # Policy profiles
        DPPolicyProfile,
        EncryptionPolicyProfile,
        TrainingPolicyProfile,
        # Key management
        KeyManagementSystem,
        KeyMetadata,
        # Evaluation gating
        EvaluationGate,
        SafetyThresholds,
        EvaluationMetrics,
        # Training pipeline
        TrainingPipeline,
        TrainingStage,
        StageConfig,
        # Observability
        ObservabilityCollector,
        RoundLatencyBreakdown,
        CompressionMetrics,
        ModelQualityMetrics,
        # Aggregation
        ResilientAggregator,
        ClientContribution,
        # Utilities
        print_production_status,
    )
except ImportError:
    pass

__all__ = [
    "EdgeClient",
    "create_client",
    "ShieldConfig",
    "Demonstration",
    "SubmissionReceipt",
    "ClientStatus",
    "VLAAdapter",
    "settings",
    
    # Production
    "OperatingEnvelope",
    "PEFTStrategy",
    "UpdatePackage",
    "ModelTargetMap",
    "TrainingMetadata",
    "SafetyStatistics",
    "ObjectiveType",
    "DPPolicyProfile",
    "EncryptionPolicyProfile",
    "TrainingPolicyProfile",
    "KeyManagementSystem",
    "KeyMetadata",
    "EvaluationGate",
    "SafetyThresholds",
    "EvaluationMetrics",
    "TrainingPipeline",
    "TrainingStage",
    "StageConfig",
    "ObservabilityCollector",
    "RoundLatencyBreakdown",
    "CompressionMetrics",
    "ModelQualityMetrics",
    "ResilientAggregator",
    "ClientContribution",
    "print_production_status",
]
