import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from tensorguard.core.crypto import N2HEEncryptor, N2HEContext, LWECiphertext, N2HEParams

def test_he_end_to_end():
    print("Testing N2HE Hardening (CSPRNG + Binary Serialization + Seeded A)...")
    
    # 1. Initialize
    encryptor = N2HEEncryptor(security_level=128)
    data = b"Hello TensorGuard!"
    
    # 2. Encrypt
    ciphertext_bytes = encryptor.encrypt(data)
    print(f"Original data size: {len(data)} bytes")
    print(f"Ciphertext size: {len(ciphertext_bytes)} bytes")
    
    # Check magic
    magic = ciphertext_bytes[:4]
    print(f"Ciphertext Magic: {magic}")
    assert magic == b"LWE2", f"Expected LWE2 (seeded), got {magic}"
    
    # 3. Decrypt
    decrypted = encryptor.decrypt(ciphertext_bytes)
    print(f"Decrypted data: {decrypted}")
    
    # Trim padding (decryption returns full block)
    assert decrypted[:len(data)] == data
    print("[OK] Encryption/Decryption PASSED")

def test_homomorphic_addition():
    print("\nTesting Homomorphic Addition...")
    params = N2HEParams()
    ctx = N2HEContext(params)
    ctx.generate_keys()
    
    # Encrypt two vectors
    v1 = np.array([10, 20, 30], dtype=np.int64)
    v2 = np.array([5, 5, 5], dtype=np.int64)
    
    ct1 = ctx.encrypt_batch(v1)
    ct2 = ctx.encrypt_batch(v2)
    
    # Add
    sum_ct = ct1 + ct2
    
    # Decrypt
    result = ctx.decrypt_batch(sum_ct)
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"Sum: {result[:3]}")
    
    assert np.all(result[:3] == (v1 + v2))
    print("[OK] Homomorphic Addition PASSED")

if __name__ == "__main__":
    try:
        test_he_end_to_end()
        test_homomorphic_addition()
        print("\nALL HE VERIFICATION TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
