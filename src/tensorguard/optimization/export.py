"""
Model Export Manager - Production Hardened

Manages model export to optimized runtimes (ONNX, TensorRT).
In production mode, requires actual dependencies and will not create dummy artifacts.
"""

import logging
import os
from typing import Any, Optional, Dict, List

from ..utils.production_gates import is_production, ProductionGateError

logger = logging.getLogger(__name__)


class ExportManager:
    """
    Manages model export to optimized runtimes (ONNX, TensorRT).

    Production behavior:
    - Requires PyTorch for ONNX export
    - Requires TensorRT for TRT compilation
    - Never creates dummy artifacts
    """

    def __init__(self):
        self.torch_available = False
        self.torch = None
        try:
            import torch

            self.torch = torch
            self.torch_available = True
        except ImportError:
            if is_production():
                raise ProductionGateError(
                    gate_name="EXPORT_TORCH",
                    message="PyTorch is required for model export in production.",
                    remediation="Install PyTorch: pip install torch",
                )
            logger.warning(
                "PyTorch not found. Export features disabled in development mode."
            )

        self.trt_available = False
        self.tensorrt = None
        try:
            import tensorrt

            self.tensorrt = tensorrt
            self.trt_available = True
        except ImportError:
            # TRT is optional even in production (GPU-specific)
            logger.info(
                "TensorRT not found. TRT compilation will be unavailable."
            )

    def is_onnx_export_available(self) -> bool:
        """Check if ONNX export is available."""
        return self.torch_available

    def is_tensorrt_available(self) -> bool:
        """Check if TensorRT compilation is available."""
        return self.trt_available

    def export_to_onnx(
        self,
        model: Any,
        input_sample: Any,
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 14,
    ) -> bool:
        """
        Exports a PyTorch model to ONNX.

        Args:
            model: PyTorch model to export
            input_sample: Sample input tensor for tracing
            output_path: Path for output ONNX file
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification
            opset_version: ONNX opset version (default 14)

        Returns:
            True if export succeeded, False otherwise

        Raises:
            ProductionGateError: If PyTorch unavailable in production
            ValueError: If model or input_sample is None in production
        """
        if not self.torch_available:
            if is_production():
                raise ProductionGateError(
                    gate_name="ONNX_EXPORT",
                    message="Cannot export to ONNX: PyTorch is not installed.",
                    remediation="Install PyTorch: pip install torch",
                )
            logger.warning(
                f"[DEV MODE] ONNX export skipped (no PyTorch). "
                f"Would export to: {output_path}"
            )
            return False

        # Validate inputs in production
        if model is None:
            if is_production():
                raise ValueError("Model cannot be None for ONNX export in production")
            logger.warning("[DEV MODE] Model is None, skipping ONNX export")
            return False

        if input_sample is None:
            if is_production():
                raise ValueError(
                    "Input sample cannot be None for ONNX export in production"
                )
            logger.warning("[DEV MODE] Input sample is None, skipping ONNX export")
            return False

        # Default names
        if input_names is None:
            input_names = ["input_ids", "attention_mask"]
        if output_names is None:
            output_names = ["logits"]
        if dynamic_axes is None:
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size", 1: "sequence"},
            }

        logger.info(f"Exporting model to ONNX: {output_path}")

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            self.torch.onnx.export(
                model,
                input_sample,
                output_path,
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

            # Verify export
            if not os.path.exists(output_path):
                raise RuntimeError(f"ONNX file was not created: {output_path}")

            file_size = os.path.getsize(output_path)
            if file_size < 100:  # Sanity check for valid ONNX file
                raise RuntimeError(
                    f"ONNX file appears invalid (size: {file_size} bytes)"
                )

            logger.info(
                f"ONNX export successful: {output_path} ({file_size / 1024 / 1024:.2f} MB)"
            )
            return True

        except Exception as e:
            logger.error(f"ONNX Export Failed: {e}")
            if is_production():
                raise
            return False

    def export_to_tensorrt(
        self,
        onnx_path: str,
        trt_path: str,
        fp16: bool = True,
        max_batch_size: int = 1,
        workspace_size_mb: int = 1024,
    ) -> bool:
        """
        Compiles an ONNX model to a TensorRT engine.

        Args:
            onnx_path: Path to input ONNX model
            trt_path: Path for output TensorRT engine
            fp16: Enable FP16 mode for faster inference
            max_batch_size: Maximum batch size for the engine
            workspace_size_mb: Workspace memory in MB

        Returns:
            True if compilation succeeded, False otherwise

        Raises:
            ProductionGateError: If TensorRT required but unavailable
            FileNotFoundError: If ONNX file doesn't exist
        """
        if not self.trt_available:
            error_msg = (
                f"Cannot compile TensorRT engine: TensorRT is not installed. "
                f"This feature requires NVIDIA TensorRT SDK."
            )
            if is_production():
                raise ProductionGateError(
                    gate_name="TENSORRT_COMPILE",
                    message=error_msg,
                    remediation="Install TensorRT from NVIDIA or skip TRT compilation.",
                )
            logger.warning(f"[DEV MODE] {error_msg}")
            return False

        # Verify ONNX file exists
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        logger.info(f"Compiling TensorRT Engine: {trt_path}")

        try:
            import tensorrt as trt

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            with trt.Builder(TRT_LOGGER) as builder:
                network_flags = 1 << int(
                    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
                )

                with builder.create_network(network_flags) as network:
                    with trt.OnnxParser(network, TRT_LOGGER) as parser:
                        # Parse ONNX model
                        with open(onnx_path, "rb") as f:
                            if not parser.parse(f.read()):
                                for idx in range(parser.num_errors):
                                    logger.error(f"TRT Parse Error: {parser.get_error(idx)}")
                                raise RuntimeError("Failed to parse ONNX model")

                        # Configure builder
                        config = builder.create_builder_config()
                        config.max_workspace_size = workspace_size_mb * 1024 * 1024

                        if fp16 and builder.platform_has_fast_fp16:
                            config.set_flag(trt.BuilderFlag.FP16)
                            logger.info("FP16 mode enabled")

                        # Build engine
                        logger.info("Building TensorRT engine (this may take a while)...")
                        engine = builder.build_engine(network, config)

                        if engine is None:
                            raise RuntimeError("Failed to build TensorRT engine")

                        # Serialize engine
                        os.makedirs(os.path.dirname(trt_path), exist_ok=True)
                        with open(trt_path, "wb") as f:
                            f.write(engine.serialize())

                        file_size = os.path.getsize(trt_path)
                        logger.info(
                            f"TensorRT engine saved: {trt_path} ({file_size / 1024 / 1024:.2f} MB)"
                        )
                        return True

        except Exception as e:
            logger.error(f"TensorRT compilation failed: {e}")
            if is_production():
                raise
            return False

    def get_export_status(self) -> Dict[str, Any]:
        """Get status of export capabilities."""
        return {
            "onnx_available": self.torch_available,
            "tensorrt_available": self.trt_available,
            "torch_version": (
                self.torch.__version__ if self.torch_available else None
            ),
            "tensorrt_version": (
                self.tensorrt.__version__ if self.trt_available else None
            ),
        }
