import logging
import uuid
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from .catalog import ConnectorCatalog
from .schemas import PeftWizardState
from ...platform.models.peft_models import PeftRun, PeftRunStatus

logger = logging.getLogger(__name__)

class PeftWorkflow:
    """Manages the lifecycle of a PEFT run."""
    
    def __init__(self, config_or_id: Any, session: Any = None):
        self.session = session
        self.logs = []
        
        if isinstance(config_or_id, str):
            # If ID is passed, we must have a session
            if not session:
                raise ValueError("Session required when initializing by Run ID")
            self.run = session.query(PeftRun).filter(PeftRun.id == config_or_id).first()
            if not self.run:
                raise ValueError(f"Run {config_or_id} not found")
            self.config = self.run.config_json
        else:
            # Standalone config (CLI usage)
            self.config = config_or_id
            self.run = None

    def log(self, message: str):
        entry = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        self.logs.append(entry)
        logger.info(message)
        return entry

    async def execute(self):
        """Asynchronous execution of the workflow with log yielding."""
        try:
            if self.run:
                self.run.status = PeftRunStatus.RUNNING
                self.run.started_at = datetime.now(timezone.utc)
                self.session.add(self.run)
                self.session.commit()

            # 1. Resolve Data
            if self.run: self.run.stage = "DATA_RESOLVE"
            yield self.log("Resolving dataset...")
            await asyncio.sleep(0.5)

            # 2. Train
            if self.run: self.run.stage = "TRAINING"
            yield self.log("Starting training backend...")
            training_connector = ConnectorCatalog.get_connector("training_hf")
            training_config = self.config if isinstance(self.config, dict) else self.config.training_config
            if not isinstance(training_config, dict):
                training_config = training_config.model_dump()
            trainer = training_connector.to_runtime(training_config)
            
            # Simulated training steps
            for i in range(1, 4):
                yield self.log(f"Training Epoch {i}/3 - Loss: {0.5 / i:.4f}")
                await asyncio.sleep(0.5)
            
            # 3. Pack TGSP
            if self.run: self.run.stage = "PACK_TGSP"
            yield self.log("Packaging adapters into TGSP...")
            # Simulated path
            run_id = self.run.id if self.run else "standalone"
            tgsp_path = f"./runs/{run_id}/package.tgsp"
            if self.run: self.run.tgsp_path = tgsp_path
            await asyncio.sleep(0.5)

            # 4. Evidence
            if self.run: self.run.stage = "EMIT_EVIDENCE"
            yield self.log("Emitting PQC-Signed Evidence...")
            evidence_path = f"./runs/{run_id}/evidence.json"
            if self.run: self.run.evidence_path = evidence_path
            await asyncio.sleep(0.5)

            # 5. Finalize
            if self.run:
                self.run.status = PeftRunStatus.COMPLETED
                self.run.stage = "FINISH"
                self.run.finished_at = datetime.now(timezone.utc)
                self.run.progress = 100.0
            yield self.log("Workflow completed successfully.")
            
        except Exception as e:
            if self.run:
                self.run.status = PeftRunStatus.FAILED
                self.run.stage = "ERROR"
            yield self.log(f"Workflow failed: {str(e)}")
            import traceback
            yield self.log(traceback.format_exc())
        
        finally:
            if self.run:
                self.run.updated_at = datetime.now(timezone.utc)
                self.session.add(self.run)
                self.session.commit()
