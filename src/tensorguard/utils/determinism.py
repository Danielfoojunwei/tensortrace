"""
Determinism Contract - Reproducibility for Training Pipelines

This module provides utilities to ensure reproducible training pipelines
when TG_DETERMINISTIC=true. Cryptographic operations remain non-deterministic
by design for security.

Usage:
    from tensorguard.utils.determinism import set_global_determinism

    # Enable deterministic mode with a seed
    set_global_determinism(seed=42)
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class DeterminismConfig:
    """Configuration for deterministic execution."""

    def __init__(
        self,
        seed: int = 42,
        deterministic_torch: bool = True,
        deterministic_cudnn: bool = True,
        warn_only: bool = False,
    ):
        self.seed = seed
        self.deterministic_torch = deterministic_torch
        self.deterministic_cudnn = deterministic_cudnn
        self.warn_only = warn_only


def is_deterministic_mode() -> bool:
    """Check if deterministic mode is enabled via environment."""
    return os.getenv("TG_DETERMINISTIC", "false").lower() == "true"


def get_determinism_seed() -> int:
    """Get the seed from environment or default."""
    try:
        return int(os.getenv("TG_SEED", "42"))
    except ValueError:
        logger.warning("Invalid TG_SEED value, using default 42")
        return 42


def set_global_determinism(
    seed: Optional[int] = None,
    deterministic_torch: bool = True,
    deterministic_cudnn: bool = True,
    log_versions: bool = True,
) -> Dict[str, Any]:
    """
    Set global determinism for reproducible training.

    This function sets seeds and configuration for:
    - Python's random module
    - NumPy
    - PyTorch (if available)
    - CUDA/cuDNN (if available)

    IMPORTANT: Cryptographic operations (key generation, signatures, etc.)
    are NOT affected by this function and remain non-deterministic by design.

    Args:
        seed: Random seed (defaults to TG_SEED env var or 42)
        deterministic_torch: Enable torch.use_deterministic_algorithms()
        deterministic_cudnn: Set cudnn.deterministic and disable benchmark
        log_versions: Log library versions for reproducibility

    Returns:
        Dict with effective settings and library versions
    """
    effective_seed = seed if seed is not None else get_determinism_seed()

    result = {
        "seed": effective_seed,
        "deterministic_torch": deterministic_torch,
        "deterministic_cudnn": deterministic_cudnn,
        "libraries": {},
    }

    logger.info("=" * 60)
    logger.info(f"Setting global determinism with seed={effective_seed}")
    logger.info("=" * 60)

    # 1. Python random
    import random

    random.seed(effective_seed)
    logger.info(f"  [OK] Python random.seed({effective_seed})")
    result["libraries"]["python"] = {
        "version": None,  # Python version is system-level
        "seed_set": True,
    }

    # 2. NumPy
    try:
        import numpy as np

        np.random.seed(effective_seed)
        result["libraries"]["numpy"] = {
            "version": np.__version__,
            "seed_set": True,
        }
        logger.info(f"  [OK] NumPy {np.__version__} seed set")
    except ImportError:
        logger.info("  [SKIP] NumPy not installed")
        result["libraries"]["numpy"] = {"available": False}

    # 3. PyTorch
    try:
        import torch

        torch.manual_seed(effective_seed)
        result["libraries"]["torch"] = {
            "version": torch.__version__,
            "seed_set": True,
            "cuda_available": torch.cuda.is_available(),
        }
        logger.info(f"  [OK] PyTorch {torch.__version__} manual_seed set")

        # CUDA seeds
        if torch.cuda.is_available():
            torch.cuda.manual_seed(effective_seed)
            torch.cuda.manual_seed_all(effective_seed)
            result["libraries"]["torch"]["cuda_seed_set"] = True
            logger.info(f"  [OK] CUDA seed set (devices: {torch.cuda.device_count()})")

            # cuDNN determinism
            if deterministic_cudnn:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                result["libraries"]["torch"]["cudnn_deterministic"] = True
                result["libraries"]["torch"]["cudnn_benchmark"] = False
                logger.info("  [OK] cuDNN deterministic mode enabled")
            else:
                logger.info("  [SKIP] cuDNN determinism disabled")

        # PyTorch deterministic algorithms
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True)
                result["libraries"]["torch"]["deterministic_algorithms"] = True
                logger.info("  [OK] torch.use_deterministic_algorithms(True)")
            except Exception as e:
                logger.warning(f"  [WARN] Could not enable deterministic algorithms: {e}")
                result["libraries"]["torch"]["deterministic_algorithms"] = False
                result["libraries"]["torch"]["deterministic_error"] = str(e)

    except ImportError:
        logger.info("  [SKIP] PyTorch not installed")
        result["libraries"]["torch"] = {"available": False}

    # 4. TensorFlow (if available)
    try:
        import tensorflow as tf

        tf.random.set_seed(effective_seed)
        result["libraries"]["tensorflow"] = {
            "version": tf.__version__,
            "seed_set": True,
        }
        logger.info(f"  [OK] TensorFlow {tf.__version__} seed set")
    except ImportError:
        pass  # TensorFlow is optional

    # 5. Environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(effective_seed)
    result["env_vars"] = {"PYTHONHASHSEED": str(effective_seed)}
    logger.info(f"  [OK] PYTHONHASHSEED={effective_seed}")

    # 6. Log versions for reproducibility
    if log_versions:
        logger.info("-" * 60)
        logger.info("Library versions for reproducibility:")
        for lib, info in result["libraries"].items():
            if info.get("version"):
                logger.info(f"  {lib}: {info['version']}")
        logger.info("-" * 60)

    logger.info("=" * 60)
    logger.info("Determinism configuration complete.")
    logger.info("NOTE: Cryptographic operations remain non-deterministic for security.")
    logger.info("=" * 60)

    return result


def get_determinism_status() -> Dict[str, Any]:
    """
    Get current determinism configuration status.

    Returns:
        Dict with current settings and library states
    """
    status = {
        "enabled": is_deterministic_mode(),
        "seed": get_determinism_seed() if is_deterministic_mode() else None,
        "libraries": {},
    }

    # Check NumPy
    try:
        import numpy as np

        # NumPy doesn't expose its current seed easily
        status["libraries"]["numpy"] = {
            "version": np.__version__,
            "available": True,
        }
    except ImportError:
        status["libraries"]["numpy"] = {"available": False}

    # Check PyTorch
    try:
        import torch

        status["libraries"]["torch"] = {
            "version": torch.__version__,
            "available": True,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            status["libraries"]["torch"]["cudnn_deterministic"] = torch.backends.cudnn.deterministic
            status["libraries"]["torch"]["cudnn_benchmark"] = torch.backends.cudnn.benchmark

    except ImportError:
        status["libraries"]["torch"] = {"available": False}

    return status


def ensure_determinism_if_enabled() -> Optional[Dict[str, Any]]:
    """
    Automatically enable determinism if TG_DETERMINISTIC=true.

    Call this at training pipeline startup to ensure reproducibility
    when the environment variable is set.

    Returns:
        Determinism config dict if enabled, None otherwise
    """
    if is_deterministic_mode():
        seed = get_determinism_seed()
        logger.info(f"TG_DETERMINISTIC=true detected, enabling determinism with seed={seed}")
        return set_global_determinism(seed=seed)

    logger.info("Deterministic mode not enabled (set TG_DETERMINISTIC=true to enable)")
    return None


# Documentation for the determinism contract
DETERMINISM_CONTRACT = """
TensorGuard Determinism Contract
================================

When TG_DETERMINISTIC=true:

1. REPRODUCIBLE (same seed = same results):
   - Model weight initialization
   - Data shuffling order
   - Dropout patterns
   - Batch normalization running stats
   - Gradient computation order (with deterministic algorithms)

2. NOT REPRODUCIBLE (intentionally):
   - Cryptographic key generation (CSPRNG)
   - Digital signatures
   - Nonces and IVs for encryption
   - Token generation

3. REQUIREMENTS for full reproducibility:
   - Same TG_SEED value
   - Same library versions (see logged versions)
   - Same hardware (GPU model can affect results)
   - Same number of workers/processes
   - Same batch size

4. LIMITATIONS:
   - Some operations have no deterministic implementation
   - Multi-GPU training may have additional non-determinism
   - Third-party libraries may not respect seeds

Example:
    export TG_DETERMINISTIC=true
    export TG_SEED=42

    # In Python:
    from tensorguard.utils.determinism import ensure_determinism_if_enabled
    ensure_determinism_if_enabled()
"""
