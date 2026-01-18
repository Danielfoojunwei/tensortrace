
import json
import hashlib

def canonical_bytes(data: dict) -> bytes:
    """
    Produce canonical bytes from a dictionary.
    Fallback to RFC 8785 JSON (sorted keys, no whitespace) since cbor2 is unavailable.
    """
    # Exclude signature if present to sign the body
    data_copy = data.copy()
    if "signature" in data_copy:
        del data_copy["signature"]
        
    return json.dumps(
        data_copy, 
        sort_keys=True, 
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')

def hash_event(data: dict) -> str:
    """Return SHA-256 hex digest of canonical event data (excluding signature)."""
    return hashlib.sha256(canonical_bytes(data)).hexdigest()
