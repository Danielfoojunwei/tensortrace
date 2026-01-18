"""
MOAI Inference Backend

SECURITY NOTE: Uses safe deserialization for weight loading.
"""

import abc
import numpy as np
import time
from typing import Dict, Any, List

from ..moai.modelpack import ModelPack
from ..utils.logging import get_logger
from ..utils.production_gates import ProductionGateError, is_production
from ..utils.serialization import safe_loads

logger = get_logger(__name__)


class MoaiBackend(abc.ABC):
    """Abstract interface for FHE inference backends."""

    @abc.abstractmethod
    def load_model(self, model_pack: ModelPack):
        """Load a ModelPack into the runtime."""
        pass

    @abc.abstractmethod
    def infer(self, ciphertext: bytes, eval_keys: bytes) -> bytes:
        """Run FHE inference."""
        pass


class TenSEALBackend(MoaiBackend):
    """
    Real FHE backend using TenSEAL (CKKS).
    Supports sequential execution of Linear layers and Polynomial Activations.
    """

    def __init__(self):
        self.active_model: ModelPack = None
        self.layers: List[Dict[str, Any]] = []

    def load_model(self, model_pack: ModelPack):
        self.active_model = model_pack
        self.layers = []

        # Unpack weights and organize into layers
        # Convention: keys are "layer_idx.type", e.g., "0.linear.weight"
        weights = model_pack.weights
        parsed_layers = {}

        for key, val in weights.items():
            parts = key.split('.')
            if not parts[0].isdigit():
                continue

            idx = int(parts[0])
            if idx not in parsed_layers:
                parsed_layers[idx] = {}

            # SECURITY: Use safe deserialization instead of pickle
            try:
                data = safe_loads(val)
                parsed_layers[idx][parts[-1]] = data
            except Exception as e:
                logger.warning(f"Failed to deserialize weight {key}: {e}")

        # Sort by index
        sorted_indices = sorted(parsed_layers.keys())
        for idx in sorted_indices:
            layer_data = parsed_layers[idx]
            # Detect type
            if "weight" in layer_data:
                self.layers.append({"type": "linear", "data": layer_data})
            elif "degree" in layer_data:
                self.layers.append({"type": "poly", "data": layer_data})

        logger.info(f"TenSEALBackend loaded model {model_pack.meta.model_id} with {len(self.layers)} layers")

    def infer(self, ciphertext: bytes, eval_keys: bytes) -> bytes:
        import tenseal as ts

        if not self.active_model:
            raise RuntimeError("No model loaded")

        try:
            # 1. Load Context & Input
            ctx = ts.context_from(eval_keys)
            enc_vec = ts.ckks_vector_from(ctx, ciphertext)

            # 2. Sequential Inference Loop
            for i, layer in enumerate(self.layers):
                ltype = layer["type"]
                data = layer["data"]

                if ltype == "linear":
                    # Linear: x @ W.T + b
                    W = data["weight"]
                    b = data.get("bias")

                    # Ensure numpy and transpose
                    if not isinstance(W, np.ndarray):
                        W = np.array(W)
                    W_t = W.T.tolist()  # TenSEAL wants list

                    enc_vec = enc_vec.matmul(W_t)

                    if b is not None:
                        if not isinstance(b, np.ndarray):
                            b = np.array(b)
                        enc_vec.add(b.tolist())

                elif ltype == "poly":
                    # Polynomial Activation: e.g. x^2
                    degree = data.get("degree", 2)
                    if degree == 2:
                        enc_vec = enc_vec.square()
                    else:
                        enc_vec = enc_vec.polyval([0] * (degree) + [1])

            # 3. Return Encrypted Result
            return enc_vec.serialize()

        except Exception as e:
            logger.error(f"FHE Inference Failed: {e}")
            raise ValueError(f"FHE Error: {e}")


class NativeBackend(MoaiBackend):
    """Placeholder for C++ MOAI runtime."""

    def __init__(self):
        if is_production():
            raise ProductionGateError(
                gate_name="MOAI_NATIVE_BACKEND",
                message="Native MOAI runtime is not available in this build.",
                remediation="Install the native MOAI runtime or configure TenSEAL backend for production.",
            )

    def load_model(self, model_pack: ModelPack):
        raise RuntimeError("Native MOAI runtime not available inside python-only environment.")

    def infer(self, ciphertext: bytes, eval_keys: bytes) -> bytes:
        raise RuntimeError("Native MOAI runtime not available.")
