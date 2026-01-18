#!/bin/bash
# Edge Agent Launcher
# Starts uploader thread and ROS 2 node

# 1. Config
API_URL="${TG_API_URL:-http://localhost:8000/api/v1/enablement}"
API_KEY="${TG_FLEET_API_KEY:-dev_key}"
DB_PATH="/var/lib/tensorguard/spool.db"

echo "Starting TensorGuard Edge Agent..."

# 2. Run Python Agent (which spawns threads for Spooler/Uploader)
# We can combine them into a single entrypoint `src/tensorguard/edge_agent/main.py`
# But for now, let's assume we run the uploader as a separate process or background it.

# Start Uploader (Background)
python -m tensorguard.edge_agent.uploader_cli \
  --db-path "$DB_PATH" \
  --url "$API_URL" \
  --key "$API_KEY" &
UPLOADER_PID=$!

# Start ROS Node (Foreground)
python -m tensorguard.edge_agent.ros2_node
NODE_EXIT=$?

kill $UPLOADER_PID
exit $NODE_EXIT
