
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from typing import Dict, Optional
import base64
import os

from .canonical import canonical_bytes

def sign_event(event_dict: dict, private_key: ed25519.Ed25519PrivateKey, key_id: str) -> dict:
    """Sign an event dictionary and return the dictionary with 'signature' field added."""
    data = canonical_bytes(event_dict)
    signature = private_key.sign(data)
    
    event_dict["signature"] = {
        "key_id": key_id,
        "alg": "ed25519",
        "sig": base64.b64encode(signature).decode('utf-8')
    }
    return event_dict

def verify_event(event_dict: dict, public_key: ed25519.Ed25519PublicKey) -> bool:
    """Verify an event dictionary's signature."""
    if "signature" not in event_dict:
        return False
        
    sig_info = event_dict["signature"]
    if sig_info.get("alg") != "ed25519":
        return False
        
    try:
        signature = base64.b64decode(sig_info["sig"])
        data = canonical_bytes(event_dict)
        public_key.verify(signature, data)
        return True
    except Exception:
        return False

def generate_keypair():
    """Generate a new Ed25519 keypair."""
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    return priv, pub

def load_private_key(path: str) -> ed25519.Ed25519PrivateKey:
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(b"-----BEGIN"):
        return serialization.load_pem_private_key(data, password=None)
    if len(data) == 32:
        return ed25519.Ed25519PrivateKey.from_private_bytes(data)
    # Default to DER if not PEM or Raw 32
    return serialization.load_der_private_key(data, password=None)

def load_public_key(path: str) -> ed25519.Ed25519PublicKey:
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(b"-----BEGIN"):
        return serialization.load_pem_public_key(data)
    if len(data) == 32:
        return ed25519.Ed25519PublicKey.from_public_bytes(data)
    return serialization.load_der_public_key(data)
