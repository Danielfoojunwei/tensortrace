"""
MOAI Serving Gateway (FastAPI)

SECURITY NOTE: Uses safe deserialization for model loading.
"""

import fastapi
from fastapi import Request, Depends, HTTPException
from pydantic import BaseModel
import uvicorn
import base64

from ..utils.logging import get_logger
from ..utils.startup_validation import validate_startup_config
from .backend import TenSEALBackend, MoaiBackend
from .auth import get_current_tenant
from ..moai.modelpack import ModelPack

logger = get_logger(__name__)

app = fastapi.FastAPI(title="TensorGuard MOAI Gateway", version="2.0.0")

# Global State (Real FHE Backend)
backend: MoaiBackend = TenSEALBackend()


class InferenceRequest(BaseModel):
    ciphertext_base64: str
    eval_keys_base64: str
    metadata: dict = {}


class InferenceResponse(BaseModel):
    result_ciphertext_base64: str
    compute_time_ms: float


@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "ok", "backend": type(backend).__name__}


@app.post("/v1/infer")
async def infer(req: InferenceRequest, tenant_id: str = Depends(get_current_tenant)):
    """
    Execute encrypted inference.
    """
    try:
        # Decode
        ct_bytes = base64.b64decode(req.ciphertext_base64)
        k_bytes = base64.b64decode(req.eval_keys_base64)

        # Infer
        import time
        t0 = time.time()
        res_bytes = backend.infer(ct_bytes, k_bytes)
        dt = (time.time() - t0) * 1000

        # Encode Response
        res_b64 = base64.b64encode(res_bytes).decode('ascii')

        return InferenceResponse(
            result_ciphertext_base64=res_b64,
            compute_time_ms=dt
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/admin/load_model")
async def load_model(file: bytes = fastapi.File(...), tenant_id: str = Depends(get_current_tenant)):
    """
    Admin endpoint to load a ModelPack binary.

    SECURITY: Uses safe deserialization (msgpack) instead of pickle.
    This prevents arbitrary code execution from malicious model files.
    """
    try:
        # Use safe deserialization from ModelPack
        pack: ModelPack = ModelPack.deserialize(file)
        backend.load_model(pack)
        return {"status": "loaded", "model_id": pack.meta.model_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ModelPack: {e}")


def start_server(port=8000):
    validate_startup_config(
        "serving",
        require_database=True,
        require_secret_key=True,
        required_dependencies=[("cryptography", "Install cryptography: pip install cryptography>=41.0")],
    )
    uvicorn.run(app, host="0.0.0.0", port=port)
