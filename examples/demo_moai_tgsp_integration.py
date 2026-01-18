"""
Demo: MOAI-TGSP Integrated Security Flow

This script demonstrates the "Double-Lock" security model:
1. MOAI: Protects Model IP (via Submodule Export) and User Privacy (via FHE).
2. TGSP: Protects Distribution (via X25519) and Provenance (via Ed25519).
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from tensorguard.moai.moai_config import MoaiConfig
from tensorguard.moai.exporter import MoaiExporter
from tensorguard.moai.keys import MoaiKeyManager
from tensorguard.tgsp.service import TGSPService

def run_integration_demo():
    print("=== MOAI-TGSP Integration Demo ===")
    
    # Setup Dirs
    work_dir = Path("demo_integration")
    work_dir.mkdir(exist_ok=True)
    
    # 1. MOAI EXPORT PHASE
    print("\n[Phase 1] MOAI Export")
    moai_cfg = MoaiConfig()
    exporter = MoaiExporter(moai_cfg)
    
    # Export a "Policy Head" ModelPack
    # In this demo, it generates mock weights for 'policy_head'
    model_pack = exporter.export(None, "vla-policy-v1", ["policy_head"])
    model_pack_path = work_dir / "policy_head.moai"
    with open(model_pack_path, "wb") as f:
        f.write(model_pack.serialize())
    print(f"  -> Generated MOAI ModelPack: {model_pack_path}")

    # 2. TGSP PACKAGING PHASE
    print("\n[Phase 2] TGSP Secure Packaging")
    
    # Generate mock keys for the demo
    producer_key = work_dir / "producer.priv"
    recipient_pub = work_dir / "gateway.pub"
    recipient_priv = work_dir / "gateway.priv"
    
    # Standalone keygen (using crypto utils)
    from tensorguard.tgsp import crypto
    
    # Ed25519 for signing
    p_key = crypto.ed25519.Ed25519PrivateKey.generate()
    with open(producer_key, "wb") as f:
        f.write(p_key.private_bytes_raw())
        
    # X25519 for encryption
    g_key = crypto.x25519.X25519PrivateKey.generate()
    with open(recipient_priv, "wb") as f:
        f.write(g_key.private_bytes_raw())
    with open(recipient_pub, "wb") as f:
        f.write(g_key.public_key().public_bytes_raw())

    tgsp_path = work_dir / "secure_inference_bundle.tgsp"
    
    # Create the package: Wrap the MOAI ModelPack as a TGSP payload
    TGSPService.create_package(
        out_path=str(tgsp_path),
        signing_key_path=str(producer_key),
        payloads=[f"moai_model:model_pack:{model_pack_path}"],
        recipients=[f"moai_gateway_01:{recipient_pub}"]
    )
    print(f"  -> Created TGSP Bundle: {tgsp_path}")

    # 3. VERIFICATION & UNPACK PHASE
    print("\n[Phase 3] Verification & Unpacking at Gateway")
    
    # Verify the bundle
    ok, msg = TGSPService.verify_package(str(tgsp_path))
    print(f"  -> Verification: {'PASS' if ok else 'FAIL'} ({msg})")
    
    # Unpack (Decrypt) the payload
    extract_dir = work_dir / "ext"
    TGSPService.decrypt_package(
        path=str(tgsp_path),
        recipient_id="moai_gateway_01",
        priv_key_path=str(recipient_priv),
        out_dir=str(extract_dir)
    )
    print(f"  -> Decrypted MOAI payload to: {extract_dir}")

    # 4. MOAI SERVING PHASE (FHE)
    print("\n[Phase 4] Loading into MOAI Gateway")
    from tensorguard.moai.modelpack import ModelPack
    from tensorguard.serving.backend import TenSEALBackend
    
    # Load the extracted ModelPack
    extracted_model_path = extract_dir / "policy_head.moai"
    with open(extracted_model_path, "rb") as f:
        final_pack = ModelPack.load(str(extracted_model_path))
    
    backend = TenSEALBackend()
    backend.load_model(final_pack)
    print(f"  -> Successfully loaded {final_pack.meta.model_id} into FHE Backend.")
    print("\nIntegration Flow: SUCCESS")

if __name__ == "__main__":
    run_integration_demo()
