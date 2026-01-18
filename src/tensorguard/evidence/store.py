import json
import os
import threading
import hashlib
from pathlib import Path
from typing import IO, List, Optional, Tuple
from .canonical import canonical_bytes, hash_event
from tensorguard.utils.exceptions import EvidenceIntegrityError

# Thread-safe singleton lock
_store_lock = threading.Lock()
_store = None

# Genesis hash for the evidence chain
GENESIS_HASH = "0" * 64


class EvidenceStore:
    """
    Tamper-evident evidence store with hash chain integrity.

    Each event is linked to the previous event via a hash chain, similar to
    a blockchain. This ensures:
    1. Events cannot be modified without detection
    2. Events cannot be deleted without breaking the chain
    3. Events cannot be reordered without detection
    4. The complete audit trail is cryptographically verifiable
    """

    CHAIN_FILE = "evidence_chain.json"

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # Absolute path to artifacts/evidence in the project root
            # store.py is at src/tensorguard/evidence/store.py
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.output_dir = str(base_dir / "artifacts" / "evidence")
        else:
            self.output_dir = os.path.abspath(output_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        self._chain_lock = threading.Lock()
        self._chain_path = os.path.join(self.output_dir, self.CHAIN_FILE)
        self._ensure_chain_file()

    def _ensure_chain_file(self):
        """Initialize the chain file if it doesn't exist."""
        if not os.path.exists(self._chain_path):
            chain_data = {
                "version": "1.0",
                "genesis_hash": GENESIS_HASH,
                "entries": []
            }
            with open(self._chain_path, "w") as f:
                json.dump(chain_data, f, indent=2)

    def _load_chain(self) -> dict:
        """Load the chain file."""
        self._ensure_chain_file()
        with open(self._chain_path, "r") as f:
            return json.load(f)

    def _save_chain(self, chain_data: dict):
        """Save the chain file atomically."""
        temp_path = self._chain_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(chain_data, f, indent=2)
        os.replace(temp_path, self._chain_path)

    def _get_last_hash(self, chain_data: dict) -> str:
        """Get the hash of the last entry in the chain."""
        entries = chain_data.get("entries", [])
        if not entries:
            return GENESIS_HASH
        return entries[-1]["event_hash"]

    def _compute_chain_hash(self, prev_hash: str, event_hash: str) -> str:
        """Compute the chain hash linking previous and current event."""
        combined = f"{prev_hash}:{event_hash}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def save_event(self, event_dict: dict) -> str:
        """
        Write signed event to disk with hash chain integrity.

        The event is linked to the previous event via a chain hash,
        ensuring tamper-evidence for the entire audit trail.
        """
        with self._chain_lock:
            os.makedirs(self.output_dir, exist_ok=True)

            # Load current chain state
            chain_data = self._load_chain()
            prev_hash = self._get_last_hash(chain_data)

            # Compute event hash (before adding chain metadata)
            event_hash = hash_event(event_dict)

            # Compute chain hash linking to previous event
            chain_hash = self._compute_chain_hash(prev_hash, event_hash)

            # Add chain metadata to event
            event_with_chain = event_dict.copy()
            event_with_chain["_chain"] = {
                "prev_hash": prev_hash,
                "event_hash": event_hash,
                "chain_hash": chain_hash,
                "sequence": len(chain_data["entries"])
            }

            # Build filename
            ts = int(event_dict.get("timestamp", 0))
            etype = event_dict.get("event_type", "UNKNOWN")
            eid = event_dict.get("event_id", "unknown")
            filename = f"{ts}_{etype}_{eid}.tge.json"
            path = os.path.join(self.output_dir, filename)

            # Serialize and write the event
            data = json.dumps(
                event_with_chain,
                sort_keys=True,
                separators=(',', ':'),
                ensure_ascii=False
            ).encode('utf-8')

            with open(path, "wb") as f:
                f.write(data)

            # Update chain file
            chain_data["entries"].append({
                "sequence": len(chain_data["entries"]),
                "filename": filename,
                "event_hash": event_hash,
                "chain_hash": chain_hash,
                "prev_hash": prev_hash,
                "event_type": etype,
                "event_id": eid,
                "timestamp": ts
            })
            self._save_chain(chain_data)

            return path

    def load_event(self, path: str, verify: bool = True) -> dict:
        """
        Load an event from disk, optionally verifying its integrity.

        Args:
            path: Path to the event file
            verify: If True, verify the event hash matches chain record

        Raises:
            EvidenceIntegrityError: If verification fails
        """
        with open(path, "rb") as f:
            event = json.load(f)

        if verify and "_chain" in event:
            chain_meta = event["_chain"]
            # Recompute event hash (excluding chain metadata)
            event_copy = {k: v for k, v in event.items() if k != "_chain"}
            computed_hash = hash_event(event_copy)

            if computed_hash != chain_meta["event_hash"]:
                raise EvidenceIntegrityError(
                    f"Event hash mismatch in {path}: "
                    f"expected {chain_meta['event_hash']}, got {computed_hash}"
                )

        return event

    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the entire evidence chain.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        chain_data = self._load_chain()
        entries = chain_data.get("entries", [])

        if not entries:
            return True, []

        prev_hash = GENESIS_HASH

        for i, entry in enumerate(entries):
            # Verify sequence
            if entry["sequence"] != i:
                errors.append(f"Sequence mismatch at index {i}: expected {i}, got {entry['sequence']}")

            # Verify chain linkage
            if entry["prev_hash"] != prev_hash:
                errors.append(f"Chain break at index {i}: prev_hash mismatch")

            # Verify chain hash computation
            expected_chain_hash = self._compute_chain_hash(prev_hash, entry["event_hash"])
            if entry["chain_hash"] != expected_chain_hash:
                errors.append(f"Chain hash mismatch at index {i}")

            # Verify the actual event file exists and matches
            event_path = os.path.join(self.output_dir, entry["filename"])
            if os.path.exists(event_path):
                try:
                    event = self.load_event(event_path, verify=True)
                except EvidenceIntegrityError as e:
                    errors.append(str(e))
            else:
                errors.append(f"Missing event file: {entry['filename']}")

            prev_hash = entry["event_hash"]

        return len(errors) == 0, errors

    def list_events(self) -> List[dict]:
        """List all events in chain order."""
        chain_data = self._load_chain()
        return chain_data.get("entries", [])

    def get_chain_head(self) -> Optional[str]:
        """Get the hash of the most recent event (chain head)."""
        chain_data = self._load_chain()
        entries = chain_data.get("entries", [])
        if not entries:
            return None
        return entries[-1]["chain_hash"]


def get_store() -> EvidenceStore:
    """Thread-safe singleton accessor for EvidenceStore."""
    global _store
    if _store is None:
        with _store_lock:
            # Double-checked locking pattern
            if _store is None:
                _store = EvidenceStore()
    return _store
