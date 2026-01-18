from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    backend: str = "hf" # hf, trl, axolotl
    method: str = "lora" # lora, qlora, adapter
    task_type: str = "sft" # sft, dpo, classification
    
    # Hyperparams
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    epochs: int = 1
    seed: int = 42
    
    # QLoRA specific
    bits: int = 4 # 4, 8, 16
    double_quant: bool = True
    
    # Model/Data refs
    model_name_or_path: str
    dataset_name_or_path: str
    output_dir: str = "./runs/latest/adapters"

class IntegrationSelector(BaseModel):
    id: str # connector id
    config_id: Optional[str] = None # existing IntegrationConfig ID
    override_config: Dict[str, Any] = {}

class PeftWizardState(BaseModel):
    """The full state sent from the UI wizard to compile/run."""
    # Steps 1-4
    training_config: TrainingConfig
    
    # Step 5: Integrations
    integrations: List[IntegrationSelector] = []
    
    # Step 6: Security & Governance
    dp_enabled: bool = False
    dp_epsilon: float = 1.0
    signing_key_path: Optional[str] = None
    policy_gates: Dict[str, Any] = {}
    
    # Step 7: Notifications
    notifications: List[IntegrationSelector] = []

class RunSummary(BaseModel):
    run_id: str
    status: str
    stage: str
    progress: float
    created_at: str

class PeftRunConfig(PeftWizardState):
    id: Optional[str] = None
    simulation: bool = True
