"""
TPM 2.0 Simulator for Hardware Attestation.

Simulates Platform Configuration Registers (PCRs) and AIK signing
to provide software-only identity claims. This is NOT a hardware
root of trust and must never be treated as attested in production
unless explicitly allowed for research use.
"""

import hashlib
import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

from ...utils.production_gates import is_production, ProductionGateError

logger = logging.getLogger(__name__)

class TPMSimulator:
    """
    Simulated Trusted Platform Module 2.0.
    
    Maintains a set of PCRs and an Attestation Identity Key (AIK).
    """
    
    def __init__(self, seed: Optional[str] = None):
        if is_production():
            allow_simulator = os.getenv("TG_ALLOW_TPM_SIMULATOR", "false").lower() == "true"
            if not allow_simulator:
                raise ProductionGateError(
                    gate_name="TPM_SIMULATOR",
                    message="TPM simulator cannot be used in production.",
                    remediation="Provision hardware-backed attestation or set TG_ALLOW_TPM_SIMULATOR=true for research only.",
                )
            logger.critical(
                "SECURITY WARNING: TPM simulator enabled in production via TG_ALLOW_TPM_SIMULATOR. "
                "Attestation claims will be marked as untrusted."
            )

        self._aik_private = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self._aik_public = self._aik_private.public_key()
        
        # Initialize PCRs (SHA-256)
        # PCR 0: BIOS/Firmware
        # PCR 1: Host Configuration
        # PCR 7: Secure Boot State
        self._pcrs: Dict[int, str] = {
            0: self._hash("TENSORGUARD_BIOS_V2"),
            1: self._hash("UBUNTU_22_04_LTS_HARDENED"),
            7: self._hash("SECURE_BOOT_ENABLED"),
        }
        
        self.boot_counter = 1
        logger.info("TPM Simulator initialized with ephemeral AIK")

    def _hash(self, data: str) -> str:
        """Helper to create SHA-256 hash in hex."""
        return hashlib.sha256(data.encode()).hexdigest()

    def extend_pcr(self, pcr_index: int, data: str):
        """Extend a PCR with new measurement."""
        if pcr_index not in self._pcrs:
            logger.warning(f"PCR {pcr_index} not initialized")
            return
            
        current_hash = bytes.fromhex(self._pcrs[pcr_index])
        new_data_hash = hashlib.sha256(data.encode()).digest()
        
        # PCR Extend: H(PCR_old || H(data))
        self._pcrs[pcr_index] = hashlib.sha256(current_hash + new_data_hash).hexdigest()
        logger.debug(f"Extended PCR {pcr_index} with {data}")

    def get_public_aik(self) -> str:
        """Return AIK public key in PEM format."""
        pem = self._aik_public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode()

    def get_quote(self, nonce: str) -> Dict[str, Any]:
        """
        Generate a signed quote over the PCRs and a nonce.
        
        Args:
            nonce: Fresh nonce from verifier to prevent replay
            
        Returns:
            Dict containing the quote structure and signature
        """
        # 1. Structure the quote data (this is what ensures integrity of state)
        quote_data = {
            "pcrs": self._pcrs,
            "nonce": nonce,
            "boot_counter": self.boot_counter,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "magic": "TG_TPM_QUOTE_V1",
            "attested": False,
            "simulator": True,
        }
        
        # 2. Canonicalize for signing
        canonical_bytes = json.dumps(quote_data, sort_keys=True).encode()
        
        # 3. Sign with AIK (RSASSA-PSS)
        signature = self._aik_private.sign(
            canonical_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return {
            "message": quote_data,
            "signature_hex": signature.hex(),
            "aik_pem": self.get_public_aik()
        }

    def verify_quote_locally(self, quote: Dict[str, Any]) -> bool:
        """Verify a quote structure (utility for testing)."""
        try:
            msg = quote["message"]
            sig = bytes.fromhex(quote["signature_hex"])
            aik_pem = quote["aik_pem"].encode()
            
            canonical_bytes = json.dumps(msg, sort_keys=True).encode()
            
            pub_key = serialization.load_pem_public_key(aik_pem)
            pub_key.verify(
                sig,
                canonical_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Quote verification failed: {e}")
            return False
