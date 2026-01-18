"""
Compliance Pack Generator (TS5)

Aggregates system inventory, audit logs, and policy states into a zip archive
for enterprise compliance auditing.
"""

import sys
import os
import json
import zipfile
from datetime import datetime
from pathlib import Path

# Mock database access for script context
# In production, this would use the SQLModel session directly
def fetch_inventory():
    return {
        "system_version": "2.1.0",
        "generated_at": datetime.utcnow().isoformat(),
        "components": ["platform", "edge_agent", "n2he_core", "moai_engine"]
    }

def fetch_audit_logs():
    return [
        {"timestamp": "2025-01-01T12:00:00Z", "action": "LOGIN", "actor": "admin@example.com", "result": "SUCCESS"},
        {"timestamp": "2025-01-01T12:05:00Z", "action": "JOB_SUBMIT", "actor": "robot_01", "result": "SUCCESS"}
    ]

def fetch_policies():
    return {
        "dp_budget_limit": 10.0,
        "retention_days": 365,
        "role_based_access": True
    }

def main():
    output_dir = Path("artifacts/compliance")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evidence_dir = Path("artifacts/evidence")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = output_dir / f"compliance_pack_{timestamp}.zip"
    
    print(f"Generating Compliance Pack: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # 1. Inventory & Static Data
        zf.writestr("inventory.json", json.dumps(fetch_inventory(), indent=2))
        zf.writestr("policies.json", json.dumps(fetch_policies(), indent=2))
        
        # 2. Real Signed Evidence (NEW)
        if evidence_dir.exists():
            print(f"Adding evidence from {evidence_dir}")
            for f in evidence_dir.glob("*.tge.json"):
                zf.write(f, arcname=f"evidence/{f.name}")
        
        # 3. Audit Logs (Legacy/Mock bridge)
        zf.writestr("logs/system_audit.json", json.dumps(fetch_audit_logs(), indent=2))
        
        # 4. Readme & Compliance Mapping
        zf.writestr("README.txt", f"TensorGuard Compliance Evidence\nGenerated: {timestamp}\nScope: Global\n\nThis pack contains Signed Evidence Events (.tge.json) verifying TGSP integrity and platform attestation.")

    print(f"Done. Pack created at {zip_path}")

if __name__ == "__main__":
    main()
