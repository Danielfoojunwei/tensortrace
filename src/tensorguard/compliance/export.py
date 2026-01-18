"""
Compliance Export Module

Bundles all compliance evidence (.tge.json files) and generates
an audit-ready package with control mappings.
"""

import os
import json
import zipfile
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Control mapping for evidence types
CONTROL_MAPPING = {
    "TGSP_BUILT": {
        "iso27001": ["A.8.24", "A.5.28"],
        "nist_csf": ["PR.DS", "PR.IP"],
        "description": "Secure Package Created with Encryption"
    },
    "TGSP_VERIFY_SUCCEEDED": {
        "iso27001": ["A.8.24", "A.5.28"],
        "nist_csf": ["PR.DS"],
        "description": "Package Signature Verified"
    },
    "ATTESTATION_VERIFIED": {
        "iso27001": ["A.14.2.2", "A.5.15"],
        "nist_csf": ["PR.AC", "DE.CM"],
        "description": "Device Attestation Passed"
    },
    "POLICY_CREATED": {
        "iso27001": ["A.8.9", "A.5.1"],
        "nist_csf": ["GV.PO"],
        "description": "Security Policy Defined"
    },
    "CERT_DISCOVERED": {
        "iso27001": ["A.8.15", "A.8.20"],
        "nist_csf": ["ID.AM"],
        "description": "Certificate Inventory Updated"
    },
    "RENEWAL_SUCCEEDED": {
        "iso27001": ["A.8.20", "A.8.24"],
        "nist_csf": ["PR.IP"],
        "description": "Certificate Renewal Completed"
    },
}


def find_evidence_files(evidence_dir: str = "./artifacts/evidence") -> List[Path]:
    """Find all .tge.json evidence files."""
    evidence_path = Path(evidence_dir)
    if not evidence_path.exists():
        logger.warning(f"Evidence directory not found: {evidence_dir}")
        return []
    
    return list(evidence_path.glob("**/*.tge.json"))


def parse_evidence_file(filepath: Path) -> Dict[str, Any]:
    """Parse an evidence JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return {
            "path": str(filepath),
            "event_type": data.get("event_type", "UNKNOWN"),
            "timestamp": data.get("timestamp", ""),
            "data": data,
        }
    except Exception as e:
        logger.error(f"Failed to parse {filepath}: {e}")
        return {"path": str(filepath), "error": str(e)}


def generate_control_report(evidence_files: List[Dict]) -> Dict[str, Any]:
    """Generate a control coverage report."""
    iso_controls = {}
    nist_controls = {}
    
    for evidence in evidence_files:
        event_type = evidence.get("event_type", "UNKNOWN")
        mapping = CONTROL_MAPPING.get(event_type, {})
        
        for ctrl in mapping.get("iso27001", []):
            if ctrl not in iso_controls:
                iso_controls[ctrl] = {"count": 0, "events": []}
            iso_controls[ctrl]["count"] += 1
            iso_controls[ctrl]["events"].append(event_type)
        
        for ctrl in mapping.get("nist_csf", []):
            if ctrl not in nist_controls:
                nist_controls[ctrl] = {"count": 0, "events": []}
            nist_controls[ctrl]["count"] += 1
            nist_controls[ctrl]["events"].append(event_type)
    
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_evidence_files": len(evidence_files),
        "iso27001_coverage": iso_controls,
        "nist_csf_coverage": nist_controls,
        "frameworks": ["ISO 27001:2022", "NIST CSF 2.0", "SOC 2", "HIPAA", "GDPR"],
    }


def export_compliance_bundle(
    evidence_dir: str = "./artifacts/evidence",
    output_path: str = "./compliance_bundle.zip"
) -> str:
    """
    Export a complete compliance bundle as a ZIP archive.
    
    Contains:
    - All .tge.json evidence files
    - control_report.json with mappings
    - manifest.json with file inventory
    """
    evidence_files = find_evidence_files(evidence_dir)
    parsed = [parse_evidence_file(f) for f in evidence_files]
    
    control_report = generate_control_report(parsed)
    
    # Build manifest
    manifest = {
        "bundle_version": "1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_directory": evidence_dir,
        "file_count": len(evidence_files),
        "files": [],
    }
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add evidence files
        for filepath in evidence_files:
            arcname = f"evidence/{filepath.name}"
            zf.write(filepath, arcname)
            
            # Compute hash for manifest
            with open(filepath, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            manifest["files"].append({
                "name": filepath.name,
                "sha256": file_hash,
            })
        
        # Add control report
        zf.writestr("control_report.json", json.dumps(control_report, indent=2))
        
        # Add manifest
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    
    logger.info(f"Compliance bundle exported to: {output_path}")
    return output_path
