"""
CLI: Incident Slicer

Creates slices from a dataset based on rules.
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "src"))
from tensorguard.enablement.robotics.slicing.incident_slicer import IncidentSlicer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Slicer")

def main():
    parser = argparse.ArgumentParser(description="TensorGuard Slicer")
    parser.add_argument("dataset_dir", help="Path to normalized dataset directory")
    parser.add_argument("--rules", title="Path to rules config json", default=None)
    args = parser.parse_args()
    
    # Mocking slice logic for now since we don't have full data stream replay in CLI yet
    slicer = IncidentSlicer()
    
    # Add a dummy manual slice based on index
    index_path = Path(args.dataset_dir) / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            meta = json.load(f)
            start = meta.get("start_time", 0)
            # Slice first 5 seconds
            slicer.add_manual_trigger(start + 2_500_000_000, "start_of_log")
            
    output = slicer.export_index()
    
    slice_file = Path(args.dataset_dir) / "slices.json"
    with open(slice_file, 'w') as f:
        json.dump(output, f, indent=2)
        
    logger.info(f"Generated {len(output)} slices to {slice_file}")

if __name__ == "__main__":
    main()
