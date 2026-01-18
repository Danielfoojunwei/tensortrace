import sys
from pathlib import Path
import numpy as np
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tensorguard.core.production import (
    UnifiedPipelineManager, 
    PipelineStage, 
    ObservabilityCollector
)

def test_hardened_workflow():
    print("Starting Hardened Pipeline Verification...")
    
    obs = ObservabilityCollector(Path("verification_metrics.jsonl"))
    manager = UnifiedPipelineManager(fleet_id="verify_fleet_01", observability=obs)
    
    # 1. Simulate a successful 7-stage loop
    print("\n[Testing Successful Workflow]")
    
    def mock_capture(data): return data
    def mock_embed(data): return np.random.rand(1, 128)
    def mock_gate(data): return "expert_active"
    def mock_peft(data): return {"layer1": np.array([0.1, 0.2])}
    def mock_shield(data): return data # Assuming output of PEFT is input to SHIELD
    
    manager.run_stage(PipelineStage.CAPTURE, mock_capture, "raw_data")
    manager.run_stage(PipelineStage.EMBED, mock_embed, "raw_data")
    manager.run_stage(PipelineStage.GATE, mock_gate, "embedding")
    manager.run_stage(PipelineStage.PEFT, mock_peft, "gate_result")
    manager.run_stage(PipelineStage.SHIELD, mock_shield, {"grad": np.array([0.1])})
    
    print(f"[OK] Workflow trace count: {len(manager.history)}")
    assert len(manager.history) == 5
    
    # 2. Simulate a failure in a critical stage (SHIELD) to test Circuit Breaker
    print("\n[Testing Circuit Breaker - SHIELD Failure]")
    
    def failing_shield(data): raise ValueError("N2HE Key Mismatch!")
    
    try:
        manager.run_stage(PipelineStage.SHIELD, failing_shield, {"grad": np.array([0.1])})
    except RuntimeError as e:
        print(f"[OK] Circuit Breaker Caught Fault: {e}")
        assert "Security Fault" in str(e)
        assert manager.safe_mode_active == True
    
    # 3. Verify Telemetry Export
    print("\n[Verifying Telemetry Export]")
    telemetry = json.loads(manager.export_telemetry())
    print(f"[OK] Safe Mode Active in Telemetry: {telemetry['safe_mode']}")
    assert telemetry['safe_mode'] == True
    
    print("\nVerification Complete: Pipeline is Hardened and Telemetric.")

if __name__ == "__main__":
    test_hardened_workflow()
