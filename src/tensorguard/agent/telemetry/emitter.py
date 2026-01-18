"""
Agent Telemetry Emitter

Emits telemetry events to the Control Plane using HMAC-authenticated requests.
Handles pipeline stage events, system metrics, and adapter swap notifications.

This module is called from edge_manager.py after each pipeline stage completes.
"""

import time
import threading
import logging
import os
import json
import hashlib
import hmac
import secrets
import requests
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class TelemetryTopic(str, Enum):
    """Standard telemetry topics."""
    STAGE = "telemetry.stage"
    SYSTEM = "telemetry.system"
    MODEL_BEHAVIOR = "telemetry.model_behavior"
    FORENSICS = "telemetry.forensics"
    HEARTBEAT = "telemetry.heartbeat"


class TelemetryEmitter:
    """
    Emits telemetry events to the Control Plane.

    Supports:
    - Pipeline stage completion events
    - System metrics (CPU, memory, disk)
    - Model behavior metrics (latency, error rate)
    - Forensics events (adapter swaps with signatures)
    """

    def __init__(
        self,
        control_plane_url: str,
        api_key: str,
        fleet_id: str,
        device_id: Optional[str] = None,
        system_metrics_interval: float = 30.0,
        enable_system_metrics: bool = True
    ):
        self.control_plane_url = control_plane_url.rstrip("/")
        self.api_key = api_key
        self.fleet_id = fleet_id
        self.device_id = device_id or os.environ.get(
            "TG_DEVICE_ID", f"device-{secrets.token_hex(4)}"
        )
        self.agent_version = os.environ.get("TG_AGENT_VERSION", "1.0.0")

        # System metrics collection
        self.system_metrics_interval = system_metrics_interval
        self.enable_system_metrics = enable_system_metrics and PSUTIL_AVAILABLE
        self._metrics_thread: Optional[threading.Thread] = None
        self._running = False

        # Event queue for batching
        self._event_queue: List[Dict[str, Any]] = []
        self._queue_lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None
        self._flush_interval = 5.0  # Flush every 5 seconds
        self._max_batch_size = 100

    def start(self):
        """Start background threads for system metrics and event flushing."""
        self._running = True

        # Start system metrics collection
        if self.enable_system_metrics:
            self._metrics_thread = threading.Thread(
                target=self._collect_system_metrics_loop,
                daemon=True,
                name="TelemetryEmitter-SystemMetrics"
            )
            self._metrics_thread.start()
            logger.info(f"System metrics collection started (interval: {self.system_metrics_interval}s)")

        # Start event flush thread
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="TelemetryEmitter-Flush"
        )
        self._flush_thread.start()
        logger.info("Telemetry emitter started")

    def stop(self):
        """Stop background threads and flush remaining events."""
        self._running = False

        # Flush any remaining events
        self._flush_events()

        if self._metrics_thread and self._metrics_thread.is_alive():
            self._metrics_thread.join(timeout=2.0)

        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2.0)

        logger.info("Telemetry emitter stopped")

    # =========================================================================
    # Public API - Event Emission
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

        Called after each stage (capture, embed, gate, peft, shield, sync, pull)
        completes.

        Args:
            stage: Stage name (capture, embed, gate, peft, shield, sync, pull)
            duration_ms: Time taken for the stage in milliseconds
            success: Whether the stage completed successfully
            error_message: Error message if stage failed
            metadata: Additional metadata about the stage execution
        """
        payload = {
            "device_id": self.device_id,
            "stage": stage,
            "duration_ms": duration_ms,
            "success": success,
            "error_message": error_message,
            "metadata": metadata or {},
        }

        self._queue_event(TelemetryTopic.STAGE, payload)

        logger.debug(f"Stage event queued: {stage} ({duration_ms:.2f}ms, success={success})")

    def emit_system_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_percent: float,
        gpu_percent: Optional[float] = None,
        gpu_memory_percent: Optional[float] = None,
        network_bytes_sent: Optional[int] = None,
        network_bytes_recv: Optional[int] = None
    ):
        """
        Emit system metrics event.

        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            disk_percent: Disk usage percentage
            gpu_percent: GPU usage percentage (if available)
            gpu_memory_percent: GPU memory usage percentage (if available)
            network_bytes_sent: Network bytes sent since last report
            network_bytes_recv: Network bytes received since last report
        """
        payload = {
            "device_id": self.device_id,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent,
            "gpu_percent": gpu_percent,
            "gpu_memory_percent": gpu_memory_percent,
            "network_bytes_sent": network_bytes_sent,
            "network_bytes_recv": network_bytes_recv,
        }

        self._queue_event(TelemetryTopic.SYSTEM, payload)

    def emit_model_behavior(
        self,
        adapter_id: str,
        inference_latency_ms: float,
        batch_size: int,
        error_rate: float,
        tokens_per_second: Optional[float] = None,
        safety_flags: Optional[List[str]] = None
    ):
        """
        Emit model behavior metrics.

        Args:
            adapter_id: ID of the adapter/model being used
            inference_latency_ms: Inference latency in milliseconds
            batch_size: Batch size for the inference
            error_rate: Error rate (0.0 - 1.0)
            tokens_per_second: Throughput in tokens per second
            safety_flags: List of safety flags triggered
        """
        payload = {
            "device_id": self.device_id,
            "adapter_id": adapter_id,
            "inference_latency_ms": inference_latency_ms,
            "batch_size": batch_size,
            "error_rate": error_rate,
            "tokens_per_second": tokens_per_second,
            "safety_flags": safety_flags or [],
        }

        self._queue_event(TelemetryTopic.MODEL_BEHAVIOR, payload)

    def emit_forensics_event(
        self,
        event_type: str,
        deployment_id: str,
        adapter_id: str,
        details: Optional[Dict[str, Any]] = None,
        pqc_signature: Optional[str] = None
    ):
        """
        Emit a forensics-grade event with optional PQC signature.

        Used for adapter swaps and other audit-critical events.

        Args:
            event_type: Type of forensics event (adapter_swap, rollback, etc.)
            deployment_id: ID of the deployment
            adapter_id: ID of the adapter involved
            details: Additional event details
            pqc_signature: PQC signature of the event (computed elsewhere)
        """
        timestamp = datetime.utcnow().isoformat()

        # If no PQC signature provided, compute a SHA256 hash as fallback
        if not pqc_signature:
            hash_input = f"{deployment_id}:{adapter_id}:{timestamp}"
            pqc_signature = f"sha256:{hashlib.sha256(hash_input.encode()).hexdigest()}"

        payload = {
            "device_id": self.device_id,
            "event_type": event_type,
            "deployment_id": deployment_id,
            "adapter_id": adapter_id,
            "details": details or {},
            "pqc_signature": pqc_signature,
            "timestamp": timestamp,
        }

        # Forensics events are high priority - flush immediately
        self._queue_event(TelemetryTopic.FORENSICS, payload, priority=10)
        self._flush_events()

        logger.info(f"Forensics event emitted: {event_type} (deployment={deployment_id})")

    def emit_heartbeat(self, status: str = "healthy"):
        """
        Emit a heartbeat event.

        Args:
            status: Agent status (healthy, degraded, etc.)
        """
        payload = {
            "device_id": self.device_id,
            "status": status,
            "agent_version": self.agent_version,
            "uptime_seconds": self._get_uptime(),
        }

        self._queue_event(TelemetryTopic.HEARTBEAT, payload)

    # =========================================================================
    # Internal - Event Queue Management
    # =========================================================================

    def _queue_event(self, topic: TelemetryTopic, payload: Dict[str, Any], priority: int = 0):
        """Add event to the queue for batched sending."""
        event = {
            "topic": topic.value,
            "timestamp_ns": int(time.time() * 1_000_000_000),
            "payload": payload,
            "priority": priority,
        }

        with self._queue_lock:
            self._event_queue.append(event)

            # Flush if queue is full
            if len(self._event_queue) >= self._max_batch_size:
                self._flush_events_internal()

    def _flush_loop(self):
        """Background loop to periodically flush events."""
        while self._running:
            time.sleep(self._flush_interval)
            self._flush_events()

    def _flush_events(self):
        """Flush queued events to the control plane."""
        with self._queue_lock:
            self._flush_events_internal()

    def _flush_events_internal(self):
        """Internal flush method (must be called with lock held)."""
        if not self._event_queue:
            return

        events = self._event_queue.copy()
        self._event_queue.clear()

        # Send in a separate thread to avoid blocking
        threading.Thread(
            target=self._send_batch,
            args=(events,),
            daemon=True
        ).start()

    def _send_batch(self, events: List[Dict[str, Any]]):
        """Send a batch of events to the control plane."""
        if not events:
            return

        try:
            # Build payload
            payload = {
                "batch_id": f"batch-{int(time.time())}-{secrets.token_hex(4)}",
                "device_info": {
                    "device_id": self.device_id,
                    "agent_version": self.agent_version,
                },
                "messages": events,
            }

            body = json.dumps(payload).encode()
            headers = self._build_headers(body)

            # Send to telemetry ingest endpoint
            url = f"{self.control_plane_url}/api/v1/telemetry/ingest"
            response = requests.post(url, data=body, headers=headers, timeout=10)

            if response.status_code in [200, 201, 202]:
                result = response.json()
                logger.debug(
                    f"Telemetry batch sent: {result.get('accepted', len(events))} accepted, "
                    f"{result.get('rejected', 0)} rejected"
                )
            else:
                logger.warning(f"Telemetry send failed: {response.status_code} {response.text}")

        except requests.exceptions.Timeout:
            logger.warning("Telemetry send timeout")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Telemetry send connection error: {e}")
        except Exception as e:
            logger.error(f"Telemetry send error: {e}")

    # =========================================================================
    # Internal - HMAC Authentication
    # =========================================================================

    def _build_headers(self, body: bytes) -> Dict[str, str]:
        """Build request headers with HMAC authentication."""
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        signature = self._compute_hmac_signature(timestamp, nonce, body)

        return {
            "Content-Type": "application/json",
            "X-TG-Fleet-Id": self.fleet_id,
            "X-TG-Timestamp": timestamp,
            "X-TG-Nonce": nonce,
            "X-TG-Signature": signature,
        }

    def _compute_hmac_signature(self, timestamp: str, nonce: str, body: bytes) -> str:
        """Compute HMAC-SHA256 signature for verify_fleet_auth."""
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{timestamp}:{nonce}:{body_hash}"
        signature = hmac.new(
            self.api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    # =========================================================================
    # Internal - System Metrics Collection
    # =========================================================================

    def _collect_system_metrics_loop(self):
        """Background loop to collect and emit system metrics."""
        last_net_io = None

        while self._running:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)

                # Memory
                memory = psutil.virtual_memory()
                memory_percent = memory.percent

                # Disk
                disk = psutil.disk_usage("/")
                disk_percent = (disk.used / disk.total) * 100

                # Network
                net_io = psutil.net_io_counters()
                bytes_sent = None
                bytes_recv = None

                if last_net_io:
                    bytes_sent = net_io.bytes_sent - last_net_io.bytes_sent
                    bytes_recv = net_io.bytes_recv - last_net_io.bytes_recv

                last_net_io = net_io

                # Emit metrics
                self.emit_system_metrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    disk_percent=disk_percent,
                    network_bytes_sent=bytes_sent,
                    network_bytes_recv=bytes_recv
                )

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")

            # Wait for next collection
            for _ in range(int(self.system_metrics_interval)):
                if not self._running:
                    break
                time.sleep(1)

    def _get_uptime(self) -> float:
        """Get process uptime in seconds."""
        try:
            if PSUTIL_AVAILABLE:
                p = psutil.Process()
                return time.time() - p.create_time()
        except Exception:
            pass
        return 0.0
