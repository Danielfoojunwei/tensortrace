"""
Proprietary Certification Engine (Enterprise).
"""

from ..utils.production_gates import ProductionGateError, is_production


def certify_artifact(artifact_id: str):
    if is_production():
        raise ProductionGateError(
            gate_name="CERTIFICATION_ENGINE",
            message="Enterprise certification engine is not available in this build.",
            remediation="Deploy with the enterprise certification package and configuration.",
        )
    raise RuntimeError(
        "Certification engine is not available in this build. "
        "Use the enterprise certification package for production."
    )
