"""
CLI: ROS 2 Ingest

Ingests a rosbag2/MCAP and creates a normalized TensorGuard dataset index.
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from tensorguard.enablement.robotics.ros2.bag_reader import RosbagReader, HAS_ROSBAGS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Ingest")

def main():
    parser = argparse.ArgumentParser(description="TensorGuard ROS2 Ingest")
    parser.add_argument("input_path", help="Path to .db3 or .mcap file")
    parser.add_argument("--output-dir", default="./dataset", help="Output directory")
    args = parser.parse_args()

    if not HAS_ROSBAGS:
        logger.error("Error: 'rosbags' library required. pip install rosbags")
        sys.exit(1)

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning: {input_path}")
    
    try:
        with RosbagReader(str(input_path)) as bag:
            meta = {
                "source": str(input_path),
                "duration": bag.get_duration(),
                "start_time": bag.get_start_time(),
                "messages": bag.get_message_count(),
                "topics": bag.get_topics()
            }
            
            logger.info("Bag Validated.")
            logger.info(f"  Duration: {meta['duration']:.2f}s")
            logger.info(f"  Messages: {meta['messages']}")
            
            # Write Index
            index_path = output_dir / "index.json"
            with open(index_path, 'w') as f:
                json.dump(meta, f, indent=2)
                
            logger.info(f"Index written to {index_path}")
            
    except Exception as e:
        logger.error(f"Ingest Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
