# Enterprise Deployment Guide (V2.0)

This guide covers the deployment of the TensorGuard Enterprise PLM suite in production environments.

## 🏗️ Deployment Architecture

The recommended production setup involves:
1.  **Kubernetes Cluster**: For orchestration of the Control Plane and Aggregator.
2.  **PostgreSQL**: High-availability database for the Control Plane.
3.  **Redis**: For caching and real-time state synchronization.
4.  **Edge Nodes**: Industrial robots or edge servers running the **Unified Agent**.

---

## 🛠️ Step 1: Pre-Deployment Setup

Ensure your environment meets the security requirements.

### Key Prerequisites
- **Python**: 3.10 or 3.11 (3.12+ recommended for Agent concurrency).
- **Network**: Port 9000 (RTPL Proxy), 8080 (Aggregator), 8000 (API).
- **Security**: Valid TLS certificates (or ACME-integrated CA).

### Installation
```bash
pip install tensorguard-enterprise[all]
make setup
```

---

## 🚀 Step 2: Launching the Control Plane

The Control Plane manages policies and audit logs.

```bash
# Set environment variables
export TG_DATABASE_URL="postgresql://user:pass@db:5432/tensorguard"
export TG_SECRET_KEY="your-production-secret"

# Initialize Database
tensorguard migrate

# Launch Server
uvicorn tensorguard.platform.main:app --host 0.0.0.0 --port 8000
```

---

## 🤖 Step 3: provisioning Edge Agents

Every edge node needs a provisioned identity.

1.  **Generate API Key**: Use the Control Plane dashboard or API.
2.  **Configure Agent**:
    ```json
    {
      "fleet_id": "production-alpha",
      "api_key": "TG_API_XXXX",
      "control_plane_url": "https://tensorguard.internal:8000",
      "storage_dir": "/var/lib/tensorguard"
    }
    ```
3.  **Start Daemon**:
    ```bash
    python -m tensorguard.agent.daemon
    ```

---

## 📈 Monitoring & Maintenance

### Audit Logs
Audit logs are stored in the primary database and should be exported to a long-term WORM (Write Once Read Many) storage for SOC 2 compliance.

### Certificate Rotation
The **Identity Manager** handles rotation automatically. Monitor for `RENEW_FAILURE` events in the audit trail.
