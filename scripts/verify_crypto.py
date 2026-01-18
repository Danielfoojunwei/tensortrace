
"""
Verification Script for TensorGuard Crypto Modules (N2HE & MOAI)
"""
import sys
import os
import logging
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CryptoVerifier")

def test_n2he():
    logger.info("--- Testing N2HE (Core) ---")
    try:
        from tensorguard.core.crypto import N2HEEncryptor, N2HEParams
        
        # 1. Initialize
        encryptor = N2HEEncryptor(security_level=128)
        logger.info("N2HE Encryptor initialized.")
        
        # 2. Encrypt/Decrypt
        data = b"Hello TensorGuard Homomorphic World!"
        ciphertext = encryptor.encrypt(data)
        logger.info(f"Encrypted {len(data)} bytes -> {len(ciphertext)} bytes (JSON/Hex)")
        
        decrypted = encryptor.decrypt(ciphertext)
        assert decrypted == data
        logger.info("Decryption successful and matches original data.")
        
    except ImportError:
        logger.error("Could not import N2HE module.")
    except Exception as e:
        logger.error(f"N2HE Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_moai():
    logger.info("--- Testing MOAI (FHE Inference) ---")
    try:
        import tenseal as ts
        from tensorguard.moai.keys import MoaiKeyManager
        from tensorguard.moai.encrypt import MoaiEncryptor, MoaiDecryptor
        from tensorguard.moai.moai_config import MoaiConfig
        
        # 1. Generate Keys
        km = MoaiKeyManager("tmp_keys")
        config = MoaiConfig()
        tid = "test_tenant"
        
        logger.info("Generating TenSEAL CKKS keys (this might take a moment)...")
        key_id, pub_ctx, sec_ctx, eval_k = km.generate_keypair(tid, config)
        logger.info(f"Keys generated: {key_id}")
        
        # 2. Encrypt
        encryptor = MoaiEncryptor(key_id, sec_ctx)
        vec = np.array([1.0, 2.0, 3.0])
        enc_vec = encryptor.encrypt_vector(vec)
        logger.info("Vector encrypted.")
        
        # 3. Decrypt
        decryptor = MoaiDecryptor(key_id, sec_ctx)
        dec_vec = decryptor.decrypt_vector(enc_vec)
        logger.info(f"Decrypted: {dec_vec}")
        
        # Basic check
        expected = np.array([1.0, 2.0, 3.0])
        if np.allclose(dec_vec, expected, atol=0.01):
            logger.info("Decryption matches original vector.")
        else:
            logger.error("Decryption mismatch!")
            
    except ImportError:
        logger.warning("TenSEAL not installed. Skipping MOAI test.")
    except Exception as e:
        logger.error(f"MOAI Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_n2he()
    print("\n")
    test_moai()
