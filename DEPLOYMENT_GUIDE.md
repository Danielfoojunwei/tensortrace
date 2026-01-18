# Deployment Guide: TensorGuard Enterprise (v2.3)

This guide covers the deployment of the entire TensorGuard fabric: the **Management Platform** (cloud/on-prem) and the **Edge Agents** (robots).

## 1. Prerequisites

### Infrastructure
- **Server**: 8+ vCPUs, 32GB RAM, Ubuntu 22.04 / Windows Server 2022.
- **Database**: PostgreSQL 14+ (or SQLite for testing).
- **Network**: Port 8000 (HTTP), 9000 (gRPC), 4173 (Frontend).

### Dependencies
- Python 3.10+
- Node.js 18+ (for Frontend build)
- NVIDIA Drivers 535+ (if using GPU acceleration)

## 2. Platform Installation (The Hub)

### Step 1: Clone & Install
```bash
git clone https://github.com/Danielfoojunwei/TensorGuardFlow
cd TensorGuardFlow
pip install -r requirements.txt
```

### Step 2: Initialize Database
```bash
# Set DB URL (defaults to sqlite:///tensorguard.db)
set DATABASE_URL=postgresql://user:pass@localhost:5432/tensorguard

# Apply Alembic migrations
alembic upgrade head
```

### Step 3: Seed Default Policies
This creates the "Default Fleet", the "CISO Admin" account, and the **Global Optimization Policy**.
```bash
python scripts/seed_db.py --admin-email admin@corp.com --optimize-policy FORCE
```

### Step 4: Start Services
Use the unified launcher:
```bash
python qa_simulation_600.py
# OR for production:
uvicorn src.tensorguard.platform.main:app --host 0.0.0.0 --port 8000
```

## 3. Edge Agent Deployment (The Robots)

### Step 1: Install SDK
On the robot (e.g., Jetson Orin):
```bash
pip install tensorguard-edge
```

### Step 2: Provision Identity
1. Go to **Mission Control** -> **Fleets** -> **Add Device**.
2. Download the `identity.json` (contains mTLS certs).
3. Place in `/etc/tensorguard/identity.json`.

### Step 3: Run Worker
```bash
tensorguard-worker --config /etc/tensorguard/identity.json
```
*Note: The worker will auto-negotiate with the server and compile local TensorRT engines upon first connection.*

## 4. Verification

1. **Dashboard**: Visit `http://localhost:8000`. You should see the **System Health** gauge at 100%.
2. **Telemetry**: Connect a test worker. Check **Mission Control** > **Efficiency Card**. You should see "Compute Speedup: ~5.4x".
3. **Logs**: Check server logs for `[POLICY] Enforcing Global 2:4 Sparsity Strategy`.

## 5. Troubleshooting

- **gRPC Connection Failed**: Check firewall on port 9000.
- **TensorRT Compilation Error**: Ensure `trtutil` is in PATH on the edge device.
- **High Latency**: Check `Efficiency Card` for "Bandwidth Saved". If low, Rand-K sparsity might be disabled in Policy.
