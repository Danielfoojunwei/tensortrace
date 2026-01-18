"""
MOAI End-to-End Demo
Simulates: Training -> Export -> Serving -> Inference
"""

import sys
import os
import numpy as np
import base64
import time

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from tensorguard.moai.moai_config import MoaiConfig
from tensorguard.moai.keys import MoaiKeyManager
from tensorguard.moai.exporter import MoaiExporter
from tensorguard.moai.encrypt import MoaiEncryptor, MoaiDecryptor


def main():
    print("=== TensorGuard MOAI Flow Demo (Real FHE) ===")
    
    # 1. Configuration & Keygen
    print("\n[1] Generating Keys (TenSEAL CKKS 8192)...")
    config = MoaiConfig()
    key_manager = MoaiKeyManager()
    # Now returns (id, public_ctx, secret_ctx, eval_keys)
    # Note: public_ctx acts as eval_keys in our simplified flow
    key_id, pk_ctx, sk_ctx, _ = key_manager.generate_keypair("tenant-A", config)
    print(f"    Key ID: {key_id}")
    print(f"    Public Context Size: {len(pk_ctx)} bytes")
    
    # 2. Export Model (Mock Training Checkpoint)
    print("\n[2] Exporting ModelPack...")
    exporter = MoaiExporter(config)
    # We mock a model path; exporter will generate weights
    # Exporter generates (128, 64) weights by default in our mock impl
    model_pack = exporter.export("mock_checkpoint.pt", "demo-model-v1", ["policy_head"])
    print(f"    ModelPack ID: {model_pack.meta.model_id}")
    
    # 3. Load Model in Server (TenSEALBackend)
    print("\n[3] Loading Model into Serving Backend...")
    # Typically this is loaded via the Gateway, but we instantiate backend directly for demo script
    from tensorguard.serving.backend import TenSEALBackend
    backend = TenSEALBackend()
    backend.load_model(model_pack)
    print("    Model loaded successfully.")
    
    # 4. Client Encryption
    print("\n[4] Client: Encrypting Input...")
    # Input must match weight shape. Weight is (128, 64). So Input is 64.
    input_vector = np.random.randn(64).astype(np.float64) # TenSEAL likes float64 usually
    
    encryptor = MoaiEncryptor(key_id, sk_ctx)
    ciphertext = encryptor.encrypt_vector(input_vector)
    print(f"    Ciphertext Size: {len(ciphertext)} bytes")
    
    # 5. Inference (Server Limit)
    print("\n[5] Server: Running Homomorphic Inference...")
    t0 = time.time()
    # Pass public context as eval keys or just context
    result_ciphertext = backend.infer(ciphertext, pk_ctx)
    dt = (time.time() - t0) * 1000
    print(f"    Inference Time: {dt:.2f}ms")
    print(f"    Result Size: {len(result_ciphertext)} bytes")
    
    # 6. Client Decryption
    print("\n[6] Client: Decrypting Result...")
    # NOTE: In real CKKS with dimension changes, we might need to know the output scale/modulus
    # But TenSEAL handles internal scale management mostly.
    decryptor = MoaiDecryptor(key_id, sk_ctx)
    result_vector = decryptor.decrypt_vector(result_ciphertext)
    
    print("\n=== Result Verification ===")
    print(f"Input Shape: {input_vector.shape}")
    print(f"Output Shape: {result_vector.shape}")
    print("First 5 values:", result_vector[:5])
    print("\n[+] Demo Complete!")

if __name__ == "__main__":
    main()
