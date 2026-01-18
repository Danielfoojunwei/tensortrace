"""
Demo: TGSP-FL Evidence & Policy Integration

This script demonstrates how TGSP provides the "Trust Layer" for Federated Learning (FL):
1. POLICY: The FL update is bound to a specific fleet policy (e.g. DP enabled).
2. EVIDENCE: The robot attaches an Evidence Report (proof of training conditions).
3. PROVENANCE: The update is signed by the robot's hardware-backed identity.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from tensorguard.tgsp.service import TGSPService
from tensorguard.tgsp import crypto as tgsp_crypto

def run_fl_policy_demo():
    print("=== TGSP-FL Evidence & Policy Integration Demo ===")
    
    # Setup Dirs
    work_dir = Path("demo_fl_trust")
    work_dir.mkdir(exist_ok=True)
    
    # 1. LOCAL TRAINING (Robot Side)
    print("\n[Phase 1] Local Training & Evidence Generation")
    
    # Simulate a gradient update
    update_data = {"weights_delta": [0.1, -0.2, 0.5], "layers": ["fc1"]}
    update_path = work_dir / "update.json"
    update_path.write_text(json.dumps(update_data))
    
    # Create an Evidence Report (Proof of local training integrity)
    # This might contain: DP epsilon used, hardware attestation hash, local data hash.
    evidence = {
        "timestamp": "2026-01-01T22:30:00Z",
        "dp_epsilon": 1.2,
        "clipping_norm": 1.0,
        "hw_attestation_token": "verified_tpm_token_0xABC",
        "data_isolation_level": "strict"
    }
    evidence_path = work_dir / "fl_evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    print(f"  -> Generated Update and Evidence Report: {evidence_path}")

    # 2. TGSP PACKAGING (Robot Side)
    print("\n[Phase 2] TGSP Wrapping (Attaching Policy & Proof)")
    
    # Robot Keys
    robot_priv = work_dir / "robot_fl.priv"
    r_key = tgsp_crypto.ed25519.Ed25519PrivateKey.generate()
    with open(robot_priv, "wb") as f:
        f.write(r_key.private_bytes_raw())
        
    # Aggregator Public Key
    agg_pub = work_dir / "aggregator_fl.pub"
    agg_key = tgsp_crypto.x25519.X25519PrivateKey.generate()
    with open(agg_pub, "wb") as f:
        f.write(agg_key.public_key().public_bytes_raw())

    tgsp_bundle = work_dir / "trusted_fl_update.tgsp"
    
    # Package the update with the Evidence Report
    # Note: We provide the 'evidence_report' argument to TGSPService
    TGSPService.create_package(
        out_path=str(tgsp_bundle),
        signing_key_path=str(robot_priv),
        payloads=[f"update:fl_gradient:{update_path}"],
        recipients=[f"federated_aggregator:{agg_pub}"],
        evidence_report=str(evidence_path)
    )
    print(f"  -> Created TGSP Update Bundle: {tgsp_bundle}")

    # 3. SERVER VERIFICATION & AUDIT (Aggregator Side)
    print("\n[Phase 3] Aggregator Verification & Audit")
    
    # Verify the bundle
    is_valid, msg = TGSPService.verify_package(str(tgsp_bundle))
    print(f"  -> TGSP Verification: {'PASS' if is_valid else 'FAIL'} ({msg})")
    
    # Unpack to inspect Evidence
    from tensorguard.tgsp.container import TGSPContainer
    from tensorguard.tgsp import spec
    
    with TGSPContainer(str(tgsp_bundle), 'r') as z:
        # Check files in the EVIDENCE directory
        evidence_files = [f for f in z.list_files() if f.startswith(spec.EVIDENCE_DIR)]
        print(f"  -> Evidence Files Found: {evidence_files}")
        
        # Read the evidence report
        report_bytes = z.read_file(evidence_files[0])
        report = json.loads(report_bytes)
        print(f"  -> Auditor Proof: DP Epsilon used = {report['dp_epsilon']}")
        
        if report['dp_epsilon'] > 2.0:
            print("  !! WARNING: Policy Violation detected in Evidence Report !!")
        else:
            print("  -> Policy Check: PASSED")

    print("\nFederated Trust Flow: SUCCESS")

if __name__ == "__main__":
    run_fl_policy_demo()
