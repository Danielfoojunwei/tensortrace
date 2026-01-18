"""
Proprietary Attestation Service (Enterprise).
"""

from ..utils.production_gates import ProductionGateError, is_production


def attest_node(node_id: str):
    if is_production():
        raise ProductionGateError(
            gate_name="ATTESTATION_SERVICE",
            message="Enterprise attestation service is not available in this build.",
            remediation="Deploy with the enterprise attestation service package and configuration.",
        )
    raise RuntimeError(
        "Attestation service is not available in this build. "
        "Use the enterprise attestation service package for production."
    )
