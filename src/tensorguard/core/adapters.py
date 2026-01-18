"""
TensorGuard VLA Adapters
Based on HintSight Technology's MoE research and MOAI (IACR 2025/991).
Incorporates Expert-Driven Aggregation (EDA) for federated robotics.
"""

import numpy as np
from typing import Dict, Any, Callable, Optional, List

from ..schemas.common import Demonstration
from ..utils.logging import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)

class VLAAdapter:
    """Base adapter for VLA models."""
    
    def __init__(self, model: Any, gradient_fn: Callable, apply_fn: Callable):
        self.model = model
        self._gradient_fn = gradient_fn
        self._apply_fn = apply_fn
    
    def compute_gradients(self, demo: Demonstration) -> Dict[str, np.ndarray]:
        """Compute gradients from demonstration."""
        try:
            return self._gradient_fn(self.model, demo)
        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            raise ValidationError(f"Invalid demonstration or model state: {e}")
    
    def apply_update(self, gradients: Dict[str, np.ndarray]) -> None:
        """Apply gradient update to model."""
        self._apply_fn(self.model, gradients)
    
    def compute_expert_gradients(self, demo: Demonstration) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute gradients split by 'Expert' category."""
        all_grads = self.compute_gradients(demo)
        experts = {"visual": {}, "language": {}, "auxiliary": {}}
        
        for k, v in all_grads.items():
            kl = k.lower()
            if any(x in kl for x in ['vision', 'encoder', 'patch']):
                experts["visual"][k] = v
            elif any(x in kl for x in ['llm', 'language', 'decoder']):
                experts["language"][k] = v
            else:
                experts["auxiliary"][k] = v
        return experts
    
    @classmethod
    def from_pi0(cls, model_path: str) -> "VLAAdapter":
        """Create adapter for Pi0 VLA."""
        from .adapters.pi0_adapter import load_pi0, pi0_gradient, pi0_apply
        return cls(load_pi0(model_path), pi0_gradient, pi0_apply)
    
    @classmethod
    def from_openvla(cls, model_path: str) -> "VLAAdapter":
        """Create adapter for OpenVLA."""
        from .adapters.openvla_adapter import load_openvla, openvla_gradient, openvla_apply
        return cls(load_openvla(model_path), openvla_gradient, openvla_apply)
    
    @classmethod
    def from_rt2(cls, model_path: str) -> "VLAAdapter":
        """Create adapter for RT-2."""
        from .adapters.rt2_adapter import load_rt2, rt2_gradient, rt2_apply
        return cls(load_rt2(model_path), rt2_gradient, rt2_apply)

class MoEAdapter(VLAAdapter):
    """
    Expert-Driven Adapter (v2.0).
    Replaces magnitude-based heuristics with Instruction-Aware Expert Gating (DGMoE).
    Addresses parameter interference in heterogeneous federated fleets.
    """
    def __init__(
        self,
        model: Any = None,
        gradient_fn: Optional[Callable] = None,
        apply_fn: Optional[Callable] = None,
        experts: List[str] = None,
    ):
        if gradient_fn is None:
            gradient_fn = self._raise_missing_gradient_fn
        if apply_fn is None:
            apply_fn = lambda m, g: None
        super().__init__(
            model=model,
            gradient_fn=gradient_fn,
            apply_fn=apply_fn,
        )
        self.experts = experts or ["visual_primary", "visual_aux", "language_semantic", "manipulation_grasp"]
        self.expert_prototypes = {
            "visual_primary": ["geometric", "shapes", "objects", "obstacles", "camera", "color", "depth"],
            "language_semantic": ["command", "intent", "goal", "instruction", "parse", "meaning"],
            "manipulation_grasp": ["gripper", "grasp", "pick", "place", "handle", "finger", "force", "torque"],
            "locomotion_base": ["move", "navigate", "base", "wheels", "collision", "path", "trajectory"],
            "fluid_pouring": ["pour", "bottle", "liquid", "tilt", "steady", "container", "cup"],
            "cleaning_wiping": ["wipe", "surface", "clean", "scrub", "pressure", "dust", "table"],
            "fastening_screwing": ["screw", "unscrew", "cap", "twist", "rotate", "thread", "bolt"]
        }

    def _raise_missing_gradient_fn(self, model, demo: Demonstration):
        raise ValidationError(
            "MoEAdapter requires a production gradient_fn implementation. "
            "Provide a real model and gradient function."
        )

    def get_expert_gate_weights(self, task_instruction: str) -> Dict[str, float]:
        """Instruction-Oriented Scene-Parsing (IOSP) simulation."""
        weights = {}
        instr = (task_instruction or "").lower()
        for exp, kws in self.expert_prototypes.items():
            relevance = sum(2.5 for kw in kws if kw in instr)
            weights[exp] = relevance + 0.1
        
        # Softmax normalize with stability
        e_x = np.exp(list(weights.values()))
        norm = e_x / (np.sum(e_x) + 1e-9)
        return dict(zip(weights.keys(), norm))

    def compute_expert_gradients(self, demo: Demonstration) -> Dict[str, Dict[str, np.ndarray]]:
        """EDA (Expert-Driven Aggregation) gradient extraction."""
        gate_weights = self.get_expert_gate_weights(demo.instruction)
        raw_grads = self.compute_gradients(demo)
        
        expert_grads = {expert: {} for expert in self.experts}
        # Simplified routing for simulation mapping blocks to experts
        routing_map = {
            "visual_primary": [0, 1, 2, 3],
            "visual_aux": [4, 5],
            "language_semantic": [6, 7],
            "manipulation_grasp": [8, 9]
        }
        
        for expert in self.experts:
            weight = gate_weights.get(expert, 0.0)
            if weight > 0.15: # Sparsity Gating (EDA)
                blocks = routing_map.get(expert, [])
                for b_idx in blocks:
                    param = f"block_{b_idx}.param"
                    if param in raw_grads:
                        expert_grads[expert][param] = raw_grads[param] * weight
        
        return expert_grads
        
class FHEExportAdapter(VLAAdapter):
    """
    Adapter for exporting specific submodules (e.g., Policy Head) for FHE inference.
    Strictly forbids exporting the entire model to prevent IP leakage.
    """
    def __init__(self, model_path: str, target_modules: List[str] = None):
        self.model_path = model_path
        self.target_modules = target_modules or ["policy_head", "visual_router"]
        self.max_params = 1_000_000  # Strict limit for FHE feasibility

    def extract_submodules(self, state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract only the allowed submodules from a full model state dict.
        """
        exported_weights = {}
        total_params = 0
        
        for k, v in state_dict.items():
            # Check if this parameter belongs to a target module
            if any(mod in k for mod in self.target_modules):
                exported_weights[k] = v
                total_params += v.size
        
        # Validation
        if total_params == 0:
            logger.warning(f"No parameters found for targets: {self.target_modules}")
        
        if total_params > self.max_params:
            raise ValidationError(f"Export exceeds FHE capacity: {total_params} > {self.max_params}")
            
        return exported_weights
