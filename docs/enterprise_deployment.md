# Enterprise Deployment Guide

This guide covers "Enterprise Table Stakes" configuration for TensorGuardFlow.

## 1. Edge Agent (TS1)
Deploy the ROS 2 Edge Agent to fleet nodes.
See `docs/edge_agent_quickstart.md`.

## 2. Identity & RBAC (TS2)
The platform now enforces strict Role-Based Access Control.
*   **Org Admin**: Full access to tenant settings.
*   **Operator**: Can submit/view jobs.
*   **Auditor**: Read-only access to Audit Logs.
*   **Service Account**: API access for robots.

## 3. Integrations (TS3)
Configure `src/tensorguard/enablement/integrations/adapters.py` to pipe events to your SOC/ChatOps.
*   **Slack**: Set `SLACK_WEBHOOK_URL` env var.
*   **PagerDuty**: Set `PD_ROUTING_KEY` for critical alerts.

## 4. Compliance (TS5)
Generate evidence packs for auditing:
```bash
python scripts/gen_compliance_pack.py
```
Outputs a ZIP file containing inventory, logs, and policy snapshots.
