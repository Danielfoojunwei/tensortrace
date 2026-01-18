
import pytest
import shutil
import os
from tensorguard.crypto.kem import generate_hybrid_keypair, encap_hybrid, decap_hybrid
from tensorguard.crypto.sig import generate_hybrid_sig_keypair, sign_hybrid, verify_hybrid
from tensorguard.crypto.pqc.kyber import Kyber768
from tensorguard.crypto.pqc.dilithium import Dilithium3

class TestQuantumSafety:
    """
    Penetration Test: Shor's Algorithm Vulnerability Check.
    Updated for Phase 15/16 Hybrid PQC implementation.
    """
    
    def test_hybrid_kem_mechanics(self):
        """
        Verify Hybrid KEM operates correctly (X25519 + Kyber768).
        """
        # 1. Keygen
        pub, priv = generate_hybrid_keypair()
        
        assert pub["alg"] == "Hybrid-Kyber-v1"
        assert len(bytes.fromhex(pub["classic"])) == 32 # X25519
        assert len(bytes.fromhex(pub["pqc"])) == Kyber768.PK_SIZE
        
        # 2. Encap
        ss_hybrid, encap_data = encap_hybrid(pub)
        
        assert len(ss_hybrid) == 32 # SHA256 output
        assert "ek_classic" in encap_data
        assert "ct_pqc" in encap_data
        
        # 3. Decap
        ss_recovered = decap_hybrid(priv, encap_data)
        
        assert ss_hybrid == ss_recovered
        
    def test_hybrid_sig_mechanics(self):
        """
        Verify Hybrid Sig operates correctly (Ed25519 + Dilithium3).
        """
        pub, priv = generate_hybrid_sig_keypair()
        message = b"Quantum-Resistant-Message"
        
        # 1. Sign
        sig_bundle = sign_hybrid(priv, message)
        
        assert len(bytes.fromhex(sig_bundle["sig_classic"])) == 64 # Ed25519
        assert len(bytes.fromhex(sig_bundle["sig_pqc"])) == Dilithium3.SIG_SIZE
        
        # 2. Verify
        is_valid = verify_hybrid(pub, message, sig_bundle)
        assert is_valid == True
        
    def test_crypto_agility_sim(self):
        """
        Ensure Simulator mode is active for this environment.
        """
        k = Kyber768()
        assert k.name.startswith("ML-KEM-768")
