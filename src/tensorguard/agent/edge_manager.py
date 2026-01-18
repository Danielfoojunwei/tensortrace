"""
Edge Agent Manager

Integrates the ROS 2 Edge Agent components (Node, Spooler, Uploader)
into the Unified Agent Daemon lifecycle.

Includes TelemetryEmitter for reporting pipeline stage completions and
system metrics to the Control Plane.
"""

import threading
import logging
import os
import time
from typing import Dict, Any, Optional

from .config_manager import ConfigManager
from .telemetry.emitter import TelemetryEmitter
from ..edge_agent.spooler import Spooler
from ..edge_agent.uploader import Uploader
from ..edge_agent.ros2_node import AgentNode, HAS_ROS2

if HAS_ROS2:
    import rclpy

logger = logging.getLogger(__name__)

class EdgeAgentManager:
    """
    Manages the ROS 2 Data Collection subsystem and telemetry emission.
    """
    def __init__(self, config: Any, config_manager: ConfigManager):
        self.config = config
        self.config_manager = config_manager
        self.running = False
        self.spooler: Optional[Spooler] = None
        self.uploader: Optional[Uploader] = None
        self.ros_node: Optional[AgentNode] = None
        self.ros_thread: Optional[threading.Thread] = None
        self.telemetry_emitter: Optional[TelemetryEmitter] = None

    def start(self):
        if self.running: return
        self.running = True
        logger.info("Starting Edge Agent Manager...")
        
        # 1. Spooler
        data_dir = getattr(self.config, 'data_dir', './storage')
        if ".." in data_dir or data_dir.startswith("/") or data_dir.startswith("\\"):
             # Local dev fallback
             data_dir = "storage"
             
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, "spool.db")
        self.spooler = Spooler(db_path)
        
        # 2. Uploader - Point to real telemetry ingestion endpoint
        base_url = self.config.control_plane_url + "/api/v1"
        api_key = self.config.api_key or os.environ.get("TG_FLEET_API_KEY")
        fleet_id = getattr(self.config, 'fleet_id', os.environ.get("TG_FLEET_ID", ""))

        if not api_key:
            logger.warning("No API Key found. Uploader disabled.")
        elif not fleet_id:
            logger.warning("No Fleet ID found. Uploader disabled.")
        else:
            self.uploader = Uploader(
                self.spooler,
                target_url=base_url + "/telemetry",
                api_key=api_key,
                fleet_id=fleet_id,
            )
            self.uploader.start()
            
        # 3. Telemetry Emitter
        if api_key and fleet_id:
            system_metrics_interval = float(os.environ.get("TG_SYSTEM_METRICS_INTERVAL", "30"))
            self.telemetry_emitter = TelemetryEmitter(
                control_plane_url=self.config.control_plane_url,
                api_key=api_key,
                fleet_id=fleet_id,
                system_metrics_interval=system_metrics_interval,
                enable_system_metrics=True
            )
            self.telemetry_emitter.start()

        # 4. ROS Node
        if HAS_ROS2:
            self.ros_thread = threading.Thread(target=self._run_ros, daemon=True)
            self.ros_thread.start()
        else:
            logger.warning("ROS 2 not found. Log collection disabled.")

    def stop(self):
        logger.info("Stopping Edge Agent Manager...")
        self.running = False

        if self.telemetry_emitter:
            self.telemetry_emitter.stop()

        if self.uploader:
            self.uploader.stop()

        if self.ros_node and HAS_ROS2 and rclpy.ok():
            self.ros_node.destroy_node()
            rclpy.shutdown()

        if self.ros_thread:
            self.ros_thread.join(timeout=2)

    def _run_ros(self):
        """Runs the ROS 2 spin loop in a thread."""
        try:
            rclpy.init()
            # Default ROS topics config (should be from self.config)
            topics = [
                {"name": "/tf", "type": "tf2_msgs/TFMessage"},
                {"name": "/odom", "type": "nav_msgs/Odometry"}
            ]
            self.ros_node = AgentNode(self.spooler, topics)
            rclpy.spin(self.ros_node)
        except Exception as e:
            logger.error(f"ROS Node error: {e}")
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    def configure(self, new_config):
        # Update uploader URL/Key if changed
        # Restart ROS node if topics changed
        pass

    # =========================================================================
    # Telemetry Emission Helpers
    # =========================================================================

    def emit_stage_event(
        self,
        stage: str,
        duration_ms: float,
        success: bool,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Emit a pipeline stage completion event.

        Call this after each stage (capture, embed, gate, peft, shield, sync, pull)
        completes.

        Args:
            stage: Stage name
            duration_ms: Time taken for the stage in milliseconds
            success: Whether the stage completed successfully
            error_message: Error message if stage failed
            metadata: Additional metadata about the stage execution
        """
        if self.telemetry_emitter:
            self.telemetry_emitter.emit_stage_event(
                stage=stage,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
                metadata=metadata
            )

    def emit_model_behavior(
        self,
        adapter_id: str,
        inference_latency_ms: float,
        batch_size: int,
        error_rate: float,
        tokens_per_second: Optional[float] = None,
        safety_flags: Optional[list] = None
    ):
        """
        Emit model behavior metrics.

        Call this after inference operations to report performance metrics.
        """
        if self.telemetry_emitter:
            self.telemetry_emitter.emit_model_behavior(
                adapter_id=adapter_id,
                inference_latency_ms=inference_latency_ms,
                batch_size=batch_size,
                error_rate=error_rate,
                tokens_per_second=tokens_per_second,
                safety_flags=safety_flags
            )

    def emit_forensics_event(
        self,
        event_type: str,
        deployment_id: str,
        adapter_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Emit a forensics-grade event for audit trail.

        Call this for adapter swaps, rollbacks, and other audit-critical events.
        """
        if self.telemetry_emitter:
            self.telemetry_emitter.emit_forensics_event(
                event_type=event_type,
                deployment_id=deployment_id,
                adapter_id=adapter_id,
                details=details
            )

    class StageTimer:
        """Context manager for timing and emitting pipeline stage events."""

        def __init__(self, manager: 'EdgeAgentManager', stage: str, metadata: Optional[Dict[str, Any]] = None):
            self.manager = manager
            self.stage = stage
            self.metadata = metadata or {}
            self.start_time: float = 0
            self.error_message: Optional[str] = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None

            if exc_val:
                self.error_message = str(exc_val)

            self.manager.emit_stage_event(
                stage=self.stage,
                duration_ms=duration_ms,
                success=success,
                error_message=self.error_message,
                metadata=self.metadata
            )

            # Don't suppress exceptions
            return False

    def stage_timer(self, stage: str, metadata: Optional[Dict[str, Any]] = None) -> 'EdgeAgentManager.StageTimer':
        """
        Create a context manager for timing and emitting stage events.

        Usage:
            with edge_manager.stage_timer("embed") as timer:
                # ... do embedding work ...
        """
        return self.StageTimer(self, stage, metadata)
