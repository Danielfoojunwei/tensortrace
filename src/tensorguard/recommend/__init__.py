"""
Proprietary Recommendation Engine (Enterprise).
"""

from ..utils.production_gates import ProductionGateError, is_production


def get_recommendations(model_id: str):
    if is_production():
        raise ProductionGateError(
            gate_name="RECOMMENDATION_ENGINE",
            message="Enterprise recommendation engine is not available in this build.",
            remediation="Deploy with the enterprise recommendation service.",
        )
    raise RuntimeError(
        "Recommendation engine is not available in this build. "
        "Use the enterprise recommendation service for production."
    )
