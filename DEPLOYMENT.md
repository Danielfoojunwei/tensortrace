# TensorGuard Deployment Guide

This guide describes how to deploy the TensorGuard Unified Trust Fabric in a production environment.

## 1. Prerequisites

*   **Python 3.10+**: Required for both Control Plane and Agent.
*   **Operating System**: Linux (Ubuntu 20.04+ recommended) or Windows Server.
*   **Network**: 
    *   Agents must have HTTPS access to the Control Plane.
    *   Control Plane needs a static IP or Domain Name.

## 2. Installation (Common)

Install the TensorGuard package on both the Server and all Edge Nodes.

```bash
# Clone the repository
git clone https://github.com/Danielfoojunwei/TensorGuardFlow
cd TensorGuardFlow

# Install Python dependencies
pip install -r requirements.txt
pip install .

# Verify installation
tensorguard --help
```

---

## 3. Control Plane Deployment (Server)

The Control Plane manages configuration, identity, and federated aggregation.

### 3.1. Database Setup
By default, the platform uses SQLite (`platform.db`). For production, configure a PostgreSQL URL in environment variables.

### 3.2. Start the Server
Run the platform server. We recommend running this behind a reverse proxy (Nginx) for SSL termination.

```bash
# Start on port 8000
tensorguard server --host 0.0.0.0 --port 8000
```

### 3.3. Verify Server
Visit `http://<server-ip>:8000/docs` to see the API Swagger UI.

---

## 4. Edge Agent Deployment

The Unified Agent runs on every robot or edge device.

### 4.1. Configuration
Create a configuration directory and file:

**Linux:** `/var/lib/tensorguard/config.json`
**Windows:** `C:\ProgramData\TensorGuard\config.json`

Example `config.json`:
```json
{
  "agent_name": "robot-alpha-001",
  "fleet_id": "fleet-sigma",
  "control_plane_url": "http://<server-ip>:8000",
  "api_key": "YOUR_FLEET_ENROLLMENT_KEY_HERE",
  "identity": {
    "enabled": true,
    "scan_interval_seconds": 3600
  },
  "network": {
    "enabled": true,
    "defense_mode": "front",
    "proxy_port": 9000,
    "target_host": "internal-controller",
    "target_port": 8080
  },
  "ml": {
    "enabled": true,
    "model_type": "pi0",
    "security_level": "medium"
  }
}
```

### 4.2. Environment Variables
Set the API Key (can be done in config or env var):

```bash
export TG_FLEET_API_KEY="your-secret-api-key"
```

### 4.3. Start the Agent Daemon
Run the agent in the background (e.g., as a Systemd service).

```bash
tensorguard agent
```

### 4.4. Verify Agent
Check the logs to confirm subsystem initialization:
```text
INFO:tensorguard.agent.daemon:Starting TensorGuard Unified Agent...
INFO:tensorguard.agent.network.guardian:Network Guardian proxy starting on port 9000
INFO:tensorguard.agent.ml.manager:ML Manager started
```

---

## 5. Production Hardening

### 5.1. Systemd Service (Linux Agent)

Create `/etc/systemd/system/tensorguard-agent.service`:

```ini
[Unit]
Description=TensorGuard Unified Agent
After=network.target

[Service]
ExecStart=/usr/local/bin/tensorguard agent
Environment="TG_FLEET_API_KEY=your-key"
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable tensorguard-agent
sudo systemctl start tensorguard-agent
```

### 5.2. Network Proxy Usage
Configure your robot's applications to use the TensorGuard local proxy for outbound traffic protection.
*   **Proxy Address:** `127.0.0.1:9000`
*   **Type:** TCP/SOCKS

### 5.3. Certificate Integration (Identity)
The Identity module will deposit renewed certificates at:
*   `/var/lib/tensorguard/certs/client.crt`
*   `/var/lib/tensorguard/certs/client.key`

Point your mTLS services (e.g., ROS2 SROS, gRPC) to these paths.
