"""
MOAI Orchestrator

Manages the lifecycle of FHE-protected models on the Edge Agent.
Responsible for:
1. Securely fetching TGSP packages.
2. Decrypting packages in-memory (SecureMemoryLoader).
3. Loading ModelPacks into the Inference Backend.

SECURITY NOTE: Uses safe deserialization instead of pickle.
"""

import logging
import io
import tarfile
import json
import numpy as np
from typing import Optional, Dict

from ...crypto.payload import PayloadDecryptor
from ...serving.backend import TenSEALBackend
from ...moai.modelpack import ModelPack
from ...tgsp.format import read_tgsp_header
from ...crypto.kem import decap_hybrid
from ...utils.serialization import safe_loads
import tempfile
import os

logger = logging.getLogger(__name__)


class SecureMemoryLoader:
    """
    Decrypts TGSP payloads directly into memory buffers.
    Ensures sensitive model weights never touch the disk.

    SECURITY: Uses safe deserialization to prevent RCE attacks.
    """

    @staticmethod
    def load_from_stream(encrypted_stream: io.BytesIO, dek: bytes) -> ModelPack:
        """
        Decrypt stream and return ModelPack.

        SECURITY: Uses ModelPack.deserialize() which uses safe msgpack
        instead of pickle to prevent arbitrary code execution.
        """
        encrypted_bytes = encrypted_stream.getvalue()

        try:
            # Try to load from TAR archive first
            with tarfile.open(fileobj=io.BytesIO(encrypted_bytes), mode="r:*") as tar:
                # Look for the model pack file (now using safe format)
                member_name = "model_pack.msgpack"
                try:
                    f = tar.extractfile(member_name)
                    if f:
                        return ModelPack.deserialize(f.read())
                except KeyError:
                    # Fallback to legacy name
                    member_name = "model_pack.bin"
                    try:
                        f = tar.extractfile(member_name)
                        if f:
                            return ModelPack.deserialize(f.read())
                    except KeyError:
                        logger.error("No model_pack file found in payload")
                        raise
        except tarfile.TarError as e:
            # Not a TAR file, try direct deserialization
            logger.debug(f"Not a TAR archive, trying direct deserialization: {e}")
            try:
                return ModelPack.deserialize(encrypted_bytes)
            except Exception:
                raise ValueError("Could not load ModelPack from memory")


class MoaiOrchestrator:
    """
    Orchestrates the MOAI inference service agent-side.
    """

    def __init__(self):
        self.backend = TenSEALBackend()
        self.active_model_id: Optional[str] = None
        self.is_ready = False

    def load_secure_package(self, package_bytes: bytes, device_private_key: bytes):
        """
        Load a TGSP package into the runtime.

        Args:
            package_bytes: The FULL TGSP container bytes.
            device_private_key: The device's private key for DEK unwrapping.
        """
        logger.info("Loading secure package into MOAI runtime...")

        # Security: Write ENCRYPTED container to temp file for parsing.
        # Plaintext is NEVER written to disk.
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(package_bytes)
            tf_path = tf.name

        try:
            # 1. Parse Header
            data = read_tgsp_header(tf_path)
            h = data["header"]

            # 2. Decrypt DEK (Hybrid Unwrapping)
            # SECURITY: Use safe deserialization for device key
            session_dek = None
            if isinstance(device_private_key, bytes):
                try:
                    # Try JSON first (preferred safe format)
                    device_sk = json.loads(device_private_key.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Fall back to safe msgpack
                    device_sk = safe_loads(device_private_key)
            else:
                device_sk = device_private_key

            for rec in data["recipients"]:
                try:
                    ss_hybrid = decap_hybrid(device_sk, rec["encap"])

                    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
                    wrapper = rec["wrapper"]
                    nonce = bytes.fromhex(wrapper["nonce"])
                    ct = bytes.fromhex(wrapper["ct"])

                    aead = ChaCha20Poly1305(ss_hybrid)
                    session_dek = aead.decrypt(nonce, ct, None)
                    break
                except Exception:
                    continue

            if not session_dek:
                raise ValueError("Could not unwrap DEK for this device")

            # 3. Setup Decryptor
            nonce_base = bytes.fromhex(h["crypto"]["nonce_base"])
            m_hash = h["hashes"]["manifest"]
            r_hash = h["hashes"]["recipients"]

            decryptor = PayloadDecryptor(session_dek, nonce_base, m_hash, r_hash)

            # 4. Stream Decrypt into Memory
            decrypted_buffer = io.BytesIO()

            with open(tf_path, "rb") as f:
                f.seek(data["payload_offset"])
                total_read = 0
                while total_read < data["payload_len"]:
                    chunk = decryptor.decrypt_chunk_from_stream(f)
                    if not chunk:
                        break
                    decrypted_buffer.write(chunk)
                    total_read += (4 + len(chunk) + 16)

            decrypted_buffer.seek(0)

            # 5. Load ModelPack (using safe deserialization)
            model_pack = SecureMemoryLoader.load_from_stream(decrypted_buffer, session_dek)

            # 6. Load Backend
            self.backend.load_model(model_pack)
            self.active_model_id = model_pack.meta.model_id
            self.is_ready = True

            logger.info(f"MOAI Runtime Ready. Active Model: {self.active_model_id}")

        except Exception as e:
            logger.error(f"Failed to load secure package: {e}")
            self.is_ready = False
            raise
        finally:
            if os.path.exists(tf_path):
                os.unlink(tf_path)

    def infer(self, ciphertext: bytes, eval_keys: bytes) -> bytes:
        """Proxy inference request to backend."""
        if not self.is_ready:
            raise RuntimeError("MoaiOrchestrator not ready (no model loaded)")

        return self.backend.infer(ciphertext, eval_keys)
