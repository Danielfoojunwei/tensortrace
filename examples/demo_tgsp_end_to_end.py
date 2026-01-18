"""
Demo: TGSP End-to-End Open Workflow
"""
import os
import subprocess
import shutil
import time
import os
import sys
import subprocess

import os

def run_cmd(args):
    print(f"Running: {' '.join(args)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run([sys.executable] + args, env=env, check=True)

def main():
    import sys
    # 0. Clean up
    if os.path.exists("demo_workdir"):
        shutil.rmtree("demo_workdir")
    os.makedirs("demo_workdir")
    
    # 1. Generate Keys
    run_cmd(["-m", "tensorguard.tgsp.cli", "keygen", "--type", "ed25519", "--name", "demo_workdir/producer"])
    run_cmd(["-m", "tensorguard.tgsp.cli", "keygen", "--type", "x25519", "--name", "demo_workdir/edge_recipient"])
    
    # 2. Mock some artifacts
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/adapter.weights", "wb") as f:
        f.write(b"MOCKED_PEFT_WEIGHT_DATA")
    
    # 3. Generate Benchmark Report (Mocked or real)
    run_cmd(["-m", "tensorguard.bench.cli", "report"])
    
    # 4. Create TGSP Package
    run_cmd(["-m", "tensorguard.tgsp.cli", "create",
            "--out", "demo_workdir/package.tgsp",
            "--producer-signing-key", "demo_workdir/producer.priv",
            "--payload", "adapter:weights:artifacts/adapter.weights",
            "--policy", "configs/policy_packs/latency-sanity-pack.yaml",
            "--recipient", "user1:demo_workdir/edge_recipient.pub",
            "--evidence-report", "artifacts/report.json"])
            
    # 5. Verify Package locally
    run_cmd(["-m", "tensorguard.tgsp.cli", "verify", "demo_workdir/package.tgsp"])
    
    print("\n--- OPEN WORKFLOW COMPLETE ---")
    print("Package created and verified locally.")
    print("Next steps in platform would be upload and release.")

if __name__ == "__main__":
    main()
