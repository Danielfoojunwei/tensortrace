"""
ML Manager - Subsystem Controller for Federated Learning

Manages the TrainingWorker and integration with the Unified Config.
Supports deployment directives for hot-swapping adapters and shadow mode.
"""

import logging
import threading
import time
from typing import Optional, Dict, Any
from ...schemas.unified_config import MLConfig, DeploymentDirective

from .worker import TrainingWorker, WorkerConfig

logger = logging.getLogger(__name__)

class MLManager:
    """
    Subsystem controller for Machine Learning tasks.
    """

    # Retry configuration
    INITIAL_RETRY_INTERVAL = 5
    MAX_RETRY_INTERVAL = 120

    def __init__(self, agent_config: 'AgentConfig', config_manager: 'ConfigManager'):
        self.config: MLConfig = agent_config.ml
        self.fleet_id = agent_config.fleet_id

        self.worker: Optional[TrainingWorker] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def configure(self, new_config: MLConfig):
        """Reconfigure ML subsystem."""
        logger.info("Reconfiguring ML Manager")
        self.config = new_config
        self._init_worker()

    def _init_worker(self):
        """Initialize the TrainingWorker."""
        worker_config = WorkerConfig(
            model_type=self.config.model_type,
            max_gradient_norm=self.config.max_gradient_norm,
            dp_epsilon=self.config.dp_epsilon,
            sparsity=self.config.sparsity,
            compression_ratio=self.config.compression_ratio
        )
        self.worker = TrainingWorker(worker_config, cid=self.fleet_id)
        
        # Determine adapter based on model_type (stub)
        # from tensorguard.core.adapters import Pi0Adapter
        # self.worker.set_adapter(Pi0Adapter())

    def start(self):
        """Start the training loop (e.g., Flower client)."""
        if not self.config.enabled:
            return
            
        self._init_worker()
        self.running = True
        
        self._thread = threading.Thread(target=self._run_flower, daemon=True)
        self._thread.start()
        logger.info("ML Manager started")

    def stop(self):
        """Stop training."""
        self.running = False
        self._stop_event.set()  # Signal to stop waiting
        # Flower client is blocking, so we can't easily stop it from outside
        # unless we terminate the process or it has a timeout.
        if self._thread:
            # self._thread.join(timeout=2.0) # Don't block on join as flwr might hang
            pass
        logger.info("ML Manager stopped")

    def _run_flower(self):
        """Run the Flower client loop with exponential backoff on errors."""
        if not self.worker:
            return

        server_address = self.config.aggregator_url.replace("http://", "").replace("https://", "") if hasattr(self.config, 'aggregator_url') else "127.0.0.1:8080"
        retry_interval = self.INITIAL_RETRY_INTERVAL

        while self.running:
            try:
                import flwr as fl
                # Run if flwr is installed
                if server_address:
                    logger.info(f"Connecting to aggregator at {server_address}")
                    fl.client.start_numpy_client(server_address=server_address, client=self.worker)
                    retry_interval = self.INITIAL_RETRY_INTERVAL  # Reset on success
            except ImportError:
                logger.warning("Flower not installed, skipping FL loop")
                break
            except Exception as e:
                logger.error(f"Flower client error: {e}")

            # Exponential backoff with event-based wait for responsive shutdown
            if self._stop_event.wait(timeout=retry_interval):
                break  # Stop event was set
            retry_interval = min(retry_interval * 2, self.MAX_RETRY_INTERVAL)

    def ingest_demonstration(self, demo_data: dict):
        """API for ingestion of local demonstrations."""
        if self.worker:
            self.worker.add_demonstration(demo_data)
            logger.info("Demonstration ingested")

    def apply_deployment_directive(self, directive: DeploymentDirective) -> Dict[str, Any]:
        """
        Apply a deployment directive from the control plane.

        Handles adapter hot-swapping and shadow mode configurations.

        Args:
            directive: DeploymentDirective from control plane with target adapter info

        Returns:
            Dict with success status and details
        """
        logger.info(
            f"Applying deployment directive: deployment_id={directive.deployment_id}, "
            f"target_adapter={directive.target_adapter_id}, "
            f"target_model_version={directive.target_model_version}, "
            f"shadow={directive.shadow}"
        )

        try:
            if directive.shadow:
                # Shadow mode: keep current adapter active, load target for comparison
                return self._apply_shadow_mode(directive)
            else:
                # Hot-swap to target adapter
                return self._hot_swap_adapter(directive)

        except Exception as e:
            logger.error(f"Failed to apply deployment directive: {e}")
            return {
                "success": False,
                "deployment_id": directive.deployment_id,
                "error": str(e)
            }

    def _apply_shadow_mode(self, directive: DeploymentDirective) -> Dict[str, Any]:
        """
        Apply shadow mode configuration.

        In shadow mode, both the current adapter and the target adapter are loaded.
        The target adapter runs in parallel but its outputs are discarded.
        This allows for A/B comparison without affecting production.
        """
        logger.info(
            f"Entering shadow mode: keeping current adapter active, "
            f"loading {directive.target_adapter_id} for shadow comparison"
        )

        # Store shadow adapter reference (stub - actual implementation would load the adapter)
        self._shadow_adapter_id = directive.target_adapter_id
        self._shadow_model_version = directive.target_model_version

        # In a full implementation, this would:
        # 1. Load the target adapter in a separate model instance
        # 2. Route inference requests to both adapters
        # 3. Log comparison metrics (latency, output similarity, etc.)
        # 4. Only return results from the primary adapter

        logger.info(
            f"Shadow mode activated: primary adapter running, "
            f"shadow adapter {directive.target_adapter_id} loaded for comparison"
        )

        return {
            "success": True,
            "deployment_id": directive.deployment_id,
            "mode": "shadow",
            "primary_adapter": getattr(self, '_current_adapter_id', 'default'),
            "shadow_adapter": directive.target_adapter_id
        }

    def _hot_swap_adapter(self, directive: DeploymentDirective) -> Dict[str, Any]:
        """
        Hot-swap to a new adapter version.

        Performs a live swap of the active adapter without stopping the training loop.
        """
        previous_adapter = getattr(self, '_current_adapter_id', 'default')

        logger.info(
            f"Hot-swapping adapter: {previous_adapter} -> {directive.target_adapter_id}"
        )

        # Store rollback information
        self._rollback_adapter_id = directive.rollback_adapter_id or previous_adapter

        # Perform the swap (stub - actual implementation would:
        # 1. Download new adapter weights if not cached
        # 2. Validate adapter compatibility with current model
        # 3. Atomically swap adapter weights in memory
        # 4. Verify model still produces valid outputs
        # )

        self._current_adapter_id = directive.target_adapter_id
        self._current_model_version = directive.target_model_version

        # Exit shadow mode if active
        if hasattr(self, '_shadow_adapter_id'):
            delattr(self, '_shadow_adapter_id')
            delattr(self, '_shadow_model_version')

        logger.info(
            f"Adapter swap complete: now running {directive.target_adapter_id} "
            f"(model version {directive.target_model_version})"
        )

        return {
            "success": True,
            "deployment_id": directive.deployment_id,
            "mode": "active",
            "previous_adapter": previous_adapter,
            "current_adapter": directive.target_adapter_id,
            "rollback_adapter": self._rollback_adapter_id
        }

    def rollback_adapter(self) -> Dict[str, Any]:
        """
        Rollback to the previous adapter version.

        Used when deployment triggers a rollback condition.
        """
        if not hasattr(self, '_rollback_adapter_id') or not self._rollback_adapter_id:
            logger.warning("No rollback adapter available")
            return {
                "success": False,
                "error": "no_rollback_available"
            }

        rollback_target = self._rollback_adapter_id
        current = getattr(self, '_current_adapter_id', 'default')

        logger.info(f"Rolling back adapter: {current} -> {rollback_target}")

        # Perform rollback (stub - actual implementation would restore previous adapter)
        self._current_adapter_id = rollback_target
        self._rollback_adapter_id = None

        logger.info(f"Rollback complete: now running {rollback_target}")

        return {
            "success": True,
            "previous_adapter": current,
            "current_adapter": rollback_target
        }
