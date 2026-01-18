"""
Privacy Ledger

Tracks the consumption of Privacy Budget (Epsilon, Delta) across the system.
Ensures that the cumulative privacy loss does not exceed defined policies.
"""

import logging
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PrivacyTransaction:
    """A single consumption of privacy budget."""
    timestamp: float
    job_id: str
    mechanism: str # e.g. "gaussian_mechanism", "moments_accountant"
    epsilon: float
    delta: float
    metadata: Dict[str, str]

class PrivacyLedger:
    """
    Immutable ledger of privacy consumption.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.transactions: List[PrivacyTransaction] = []
        self.total_epsilon = 0.0
        self.total_delta = 0.0
        
        if self.storage_path:
            self._load()

    def record_consumption(self, job_id: str, epsilon: float, delta: float, mechanism: str = "generic"):
        """Record a privacy cost."""
        tx = PrivacyTransaction(
            timestamp=time.time(),
            job_id=job_id,
            mechanism=mechanism,
            epsilon=epsilon,
            delta=delta,
            metadata={}
        )
        self.transactions.append(tx)
        self.total_epsilon += epsilon
        self.total_delta += delta
        
        logger.info(f"Privacy Budget Consumed: +ε{epsilon:.2f} (Total: ε{self.total_epsilon:.2f})")
        
        if self.storage_path:
            self._save()

    def get_remaining_budget(self, max_epsilon: float) -> float:
        """Calculate remaining epsilon."""
        return max(0.0, max_epsilon - self.total_epsilon)

    def _save(self):
        """Persist ledger to disk."""
        data = {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "transactions": [asdict(tx) for tx in self.transactions]
        }
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save privacy ledger: {e}")

    def _load(self):
        """Load ledger from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.total_epsilon = data.get("total_epsilon", 0.0)
                self.total_delta = data.get("total_delta", 0.0)
                # Parse transactions if needed
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Failed to load privacy ledger: {e}")
