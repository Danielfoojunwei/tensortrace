# Edge Agent Quickstart (ROS 2)

This guide explains how to deploy the TensorGuard Edge Agent on a ROS 2 robot.

## Components
1.  **Node**: Subscribes to `/tf`, `/odom`, `/joint_states` (configurable).
2.  **Spooler**: Writes messages to local SQLite buffer (`spool.db`).
3.  **Uploader**: Asynchronously pushes buffer to Enablement Platform.

## Prerequisites
*   ROS 2 (Humble/Iron/Rolling) installed and sourced.
*   Python 3.10+
*   TensorGuardFlow installed (`pip install .`)

## Configuration
Edit `src/tensorguard/edge_agent/ros2_node.py` (or provide config file in future) to select topics.

## Running

```bash
# 1. Source ROS 2
source /opt/ros/humble/setup.bash

# 2. Set Credentials
export TG_FLEET_API_KEY="your-api-key"
export TG_API_URL="http://your-platform:8000/api/v1/enablement"

# 3. Run
bash scripts/edge_agent_run.sh
```

## Troubleshooting
*   **Missing Imports**: Ensure `rclpy` is in your PYTHONPATH.
*   **Connection Error**: Check `TG_API_URL`. The spooler will buffer data until connection is restored.
