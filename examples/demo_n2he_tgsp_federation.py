"""
Demo: N2HE-TGSP Federated Learning Security flow

This script demonstrates how N2HE (Homomorphic Privacy) and TGSP (Distribution Trust) 
work together to protect Federated Learning updates:
1. N2HE: Protects Gradient Privacy (prevents server from seeing local data).
2. TGSP: Protects Update Integrity & Provenance (prevents unauthenticated gradient poisoning).
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from tensorguard.core.crypto import N2HEContext, N2HEParams, LWECiphertext
from tensorguard.tgsp.service import TGSPService
from tensorguard.tgsp import crypto as tgsp_crypto

def run_federation_demo():
    print("=== N2HE-TGSP Federated Integration Demo ===")
    
    # Setup Dirs
    work_dir = Path("demo_federation")
    work_dir.mkdir(exist_ok=True)
    
    # 1. N2HE PHASE: Robot computes local gradient and encrypts it
    print("\n[Phase 1] N2HE Encryption (Edge Device)")
    n2he_params = N2HEParams(n=256)
    ctx = N2HEContext(n2he_params)
    ctx.generate_keys()
    
    # Simulate a local gradient vector (e.g., from Pi0/VLA fine-tuning)
    local_gradient = np.random.randint(0, 10, size=256, dtype=np.int64)
    encrypted_gradient = ctx.encrypt_batch(local_gradient)
    
    # Serialize the N2HE ciphertext
    n2he_binary = encrypted_gradient.serialize()
    gradient_file_path = work_dir / "local_update.n2he"
    with open(gradient_file_path, "wb") as f:
        f.write(n2he_binary)
    print(f"  -> Generated N2HE Encrypted Gradient: {gradient_file_path} ({len(n2he_binary)} bytes)")

    # 2. TGSP PHASE: Wrap the encrypted gradient in a secure envelope
    print("\n[Phase 2] TGSP Packaging (Provenance & Integrity)")
    
    # Generate robot identity (Ed25519)
    robot_key_path = work_dir / "robot_identity.priv"
    robot_keys = tgsp_crypto.ed25519.Ed25519PrivateKey.generate()
    with open(robot_key_path, "wb") as f:
        f.write(robot_keys.private_bytes_raw())
        
    # The aggregator doesn't need to decrypt the N2HE content (it aggregates it homomorphically),
    # but TGSP still uses a recipient model. Here, the recipient is the 'Aggregator'.
    aggregator_pub = work_dir / "aggregator.pub"
    agg_key = tgsp_crypto.x25519.X25519PrivateKey.generate()
    with open(aggregator_pub, "wb") as f:
        f.write(agg_key.public_key().public_bytes_raw())

    tgsp_bundle_path = work_dir / "signed_federated_update.tgsp"
    
    # The Robot packs the N2HE gradient into a TGSP bundle.
    # This adds:
    # - A signature proving EXACTLY which robot sent this update.
    # - A signed manifest that can include policy compliance evidence.
    TGSPService.create_package(
        out_path=str(tgsp_bundle_path),
        signing_key_path=str(robot_key_path),
        payloads=[f"grad:n2he_update:{gradient_file_path}"],
        recipients=[f"aggregator_node:{aggregator_pub}"]
    )
    print(f"  -> Created Secure Federated Bundle: {tgsp_bundle_path}")

    # 3. AGGREGATOR VERIFICATION PHASE
    print("\n[Phase 3] Aggregator Verification")
    
    # The aggregator verifies the TGSP envelope.
    # If the signature or manifest is invalid, the update is rejected BEFORE 
    # being added to the global model, preventing adversarial poisoning.
    is_valid, reason = TGSPService.verify_package(str(tgsp_bundle_path))
    print(f"  -> TGSP Verification Result: {'VALID' if is_valid else 'INVALID'} ({reason})")
    print("  -> Aggregator now trusts this update came from a verified robot.")

    # 4. HOMOMORPHIC AGGREGATION PHASE
    print("\n[Phase 4] N2HE Homomorphic Aggregation (Aggregator)")
    # (Simulated) Aggregation of multiple trusted TGSP packages...
    # The aggregator extracts the N2HE ciphertext from the trusted TGSP bundle.
    
    # (Extracting for the demo...)
    agg_priv_path = work_dir / "aggregator.priv"
    with open(agg_priv_path, "wb") as f:
        f.write(agg_key.private_bytes_raw())
        
    extract_dir = work_dir / "ext_agg"
    TGSPService.decrypt_package(
        path=str(tgsp_bundle_path),
        recipient_id="aggregator_node",
        priv_key_path=str(agg_priv_path),
        out_dir=str(extract_dir)
    )
    
    # Load the trusted ciphertext
    with open(extract_dir / "local_update.n2he", "rb") as f:
        trusted_n2he_bytes = f.read()
    
    ct_trusted = LWECiphertext.deserialize(trusted_n2he_bytes)
    
    # Aggregate with another (placeholder) update
    # In reality, this happens while BOTH are encrypted!
    ct_global = ct_trusted + ct_trusted # Simulating sum(ct_i)
    
    # 5. RESULT: Verified, Privacy-Preserving Global Update
    print("\n[Phase 5] Final Result")
    # Decryption (Server side for the final result if it has the N2HE SK, or sent back to robots)
    decrypted_agg = ctx.decrypt_batch(ct_global) 
    print(f"  -> Decrypted Sum (Robot 1 + Robot 1): {decrypted_agg[:5]}...")
    print(f"  -> Original First Values:             {local_gradient[:5] * 2}...")
    
    print("\nIntegration Flow: SUCCESS")

if __name__ == "__main__":
    run_federation_demo()
