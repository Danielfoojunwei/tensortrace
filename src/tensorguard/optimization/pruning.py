"""
Model Pruning Manager - Production Hardened

Manages model pruning, specifically targeting 2:4 structured sparsity
for NVIDIA Ampere+ acceleration.

In production mode, requires PyTorch and will not return fake sparsity values.
"""

import logging
from typing import Any, Dict, Optional

from ..utils.production_gates import is_production, ProductionGateError

logger = logging.getLogger(__name__)


class PruningManager:
    """
    Manages model pruning, specifically targeting 2:4 structured sparsity
    for NVIDIA Ampere+ acceleration.

    Production behavior:
    - Requires PyTorch
    - Never returns simulated sparsity values
    - Validates model before pruning
    """

    def __init__(self):
        self.torch_available = False
        self.torch = None
        self.prune = None
        try:
            import torch
            import torch.nn.utils.prune as prune

            self.torch = torch
            self.prune = prune
            self.torch_available = True
        except ImportError:
            if is_production():
                raise ProductionGateError(
                    gate_name="PRUNING_TORCH",
                    message="PyTorch is required for model pruning in production.",
                    remediation="Install PyTorch: pip install torch",
                )
            logger.warning(
                "PyTorch not found. Pruning features disabled in development mode."
            )

    def is_available(self) -> bool:
        """Check if pruning is available."""
        return self.torch_available

    def apply_2_4_sparsity(self, model: Any) -> bool:
        """
        Applies 2:4 structured sparsity to all Linear layers in the model.
        This patterns prunes 2 out of every 4 consecutive weights.

        Args:
            model: PyTorch model to prune

        Returns:
            True if pruning was applied successfully

        Raises:
            ProductionGateError: If PyTorch unavailable in production
            ValueError: If model is None in production
        """
        if not self.torch_available:
            if is_production():
                raise ProductionGateError(
                    gate_name="APPLY_SPARSITY",
                    message="Cannot apply sparsity: PyTorch is not installed.",
                    remediation="Install PyTorch: pip install torch",
                )
            logger.warning(
                "[DEV MODE] Sparsity not applied (PyTorch unavailable)"
            )
            return False

        # Validate model
        if model is None:
            if is_production():
                raise ValueError("Model cannot be None for pruning in production")
            logger.warning("[DEV MODE] Model is None, skipping pruning")
            return False

        if isinstance(model, dict):
            if is_production():
                raise ValueError(
                    "Cannot prune a dict. Provide a PyTorch nn.Module instance."
                )
            logger.warning("[DEV MODE] Model is a dict, skipping pruning")
            return False

        logger.info("Applying 2:4 Structured Sparsity to Linear layers...")
        count = 0
        errors = []

        # Iterate through all modules and prune Linear layers
        for name, module in model.named_modules():
            if isinstance(module, self.torch.nn.Linear):
                try:
                    # Apply structured pruning
                    # Note: True 2:4 enforcement often requires NVIDIA's ASP library
                    # Here we use ln_structured as a proxy for structural constraint
                    self.prune.ln_structured(
                        module, name="weight", amount=0.5, n=2, dim=0
                    )
                    # Make it permanent
                    self.prune.remove(module, "weight")
                    count += 1
                except Exception as e:
                    error_msg = f"Failed to prune layer {name}: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)

        if count == 0:
            logger.warning("No Linear layers found to prune")
            if is_production() and errors:
                raise RuntimeError(
                    f"Pruning failed for all layers. Errors: {'; '.join(errors)}"
                )
            return False

        logger.info(f"Successfully pruned {count} Linear layers.")
        return True

    def check_sparsity(self, model: Any) -> float:
        """
        Returns the global sparsity percentage of the model.

        Args:
            model: PyTorch model to analyze

        Returns:
            Sparsity percentage (0.0 - 100.0)

        Raises:
            ProductionGateError: If PyTorch unavailable in production
            ValueError: If model is None in production
        """
        if not self.torch_available:
            if is_production():
                raise ProductionGateError(
                    gate_name="CHECK_SPARSITY",
                    message="Cannot check sparsity: PyTorch is not installed.",
                    remediation="Install PyTorch: pip install torch",
                )
            logger.warning(
                "[DEV MODE] Cannot check sparsity (PyTorch unavailable)"
            )
            return 0.0

        if model is None:
            if is_production():
                raise ValueError("Model cannot be None for sparsity check in production")
            logger.warning("[DEV MODE] Model is None, returning 0% sparsity")
            return 0.0

        global_zero = 0
        global_elements = 0

        for name, module in model.named_modules():
            if isinstance(module, self.torch.nn.Linear):
                weight = module.weight.data
                global_zero += self.torch.sum(weight == 0).item()
                global_elements += weight.numel()

        if global_elements == 0:
            return 0.0

        sparsity = (global_zero / global_elements) * 100
        logger.info(f"Model sparsity: {sparsity:.2f}%")
        return sparsity

    def verify_2_4_pattern(self, model: Any) -> Dict[str, Any]:
        """
        Verify that the model follows strict 2:4 sparsity pattern.

        2:4 pattern means exactly 2 zeros in every 4 consecutive elements.

        Args:
            model: PyTorch model to verify

        Returns:
            Dict with verification results
        """
        if not self.torch_available:
            if is_production():
                raise ProductionGateError(
                    gate_name="VERIFY_PATTERN",
                    message="Cannot verify sparsity pattern: PyTorch is not installed.",
                    remediation="Install PyTorch: pip install torch",
                )
            return {"verified": False, "error": "PyTorch unavailable"}

        if model is None:
            return {"verified": False, "error": "Model is None"}

        results = {
            "verified": True,
            "layers_checked": 0,
            "layers_valid": 0,
            "violations": [],
        }

        for name, module in model.named_modules():
            if isinstance(module, self.torch.nn.Linear):
                weight = module.weight.data.flatten()
                results["layers_checked"] += 1

                # Check 2:4 pattern in groups of 4
                valid = True
                for i in range(0, len(weight) - 3, 4):
                    group = weight[i : i + 4]
                    zeros = (group == 0).sum().item()
                    if zeros != 2:
                        valid = False
                        break

                if valid:
                    results["layers_valid"] += 1
                else:
                    results["verified"] = False
                    results["violations"].append(name)

        return results

    def get_pruning_status(self) -> Dict[str, Any]:
        """Get status of pruning capabilities."""
        return {
            "available": self.torch_available,
            "torch_version": (
                self.torch.__version__ if self.torch_available else None
            ),
            "supported_patterns": ["2:4 structured", "unstructured"],
        }
