#!/usr/bin/env python3
"""
Identity Demo Script - End-to-End Certificate Lifecycle Demo

Demonstrates:
1. Tenant + Fleet setup
2. Agent enrollment
3. Certificate scan
4. Policy creation (200-day preset)
5. Renewal job execution
6. Audit log verification
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta

# Configuration
BASE_URL = os.environ.get("TG_API_URL", "http://localhost:8000/api/v1")
ADMIN_EMAIL = "admin@demo.tensorguard.io"
ADMIN_PASS = "Demo-Password-123!"
TENANT_NAME = "TensorGuard Demo"
FLEET_NAME = "production-k8s"

def print_step(step: int, message: str):
    """Print step with formatting."""
    print(f"\n{'='*60}")
    print(f"Step {step}: {message}")
    print('='*60)

def api_call(method: str, path: str, token: str = None, data: dict = None):
    """Make API call with optional auth."""
    url = f"{BASE_URL}{path}"
    headers = {}
    
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    if method == "GET":
        response = requests.get(url, headers=headers, params=data, timeout=30)
    elif method == "POST":
        headers["Content-Type"] = "application/json"
        response = requests.post(url, headers=headers, json=data, timeout=30)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    if not response.ok:
        print(f"  [ERROR] Status: {response.status_code} - {response.text}")
        return None
    
    return response.json()

def main():
    print("\n" + "* "*20)
    print("TensorGuard Machine Identity Guard - Demo")
    print("* "*20)
    
    token = None
    
    # Step 1: Initialize Tenant
    print_step(1, "Initialize Tenant")
    
    result = api_call(
        "POST",
        f"/onboarding/init?name={TENANT_NAME}&admin_email={ADMIN_EMAIL}&admin_pass={ADMIN_PASS}"
    )
    
    if result:
        print(f"  [OK] Tenant created: {result.get('name')} (ID: {result.get('id')})")
    else:
        print("  [WARN] Tenant may already exist, trying login...")
    
    # Step 2: Login
    print_step(2, "Authenticate")
    
    result = api_call("POST", "/auth/token", data={
        "username": ADMIN_EMAIL,
        "password": ADMIN_PASS,
    })
    
    if result:
        token = result.get("access_token")
        print(f"  [OK] Logged in successfully")
    else:
        print("  [ERROR] Login failed")
        return 1
    
    # Step 3: Create Fleet
    print_step(3, "Create Fleet")
    
    result = api_call("POST", f"/fleets?name={FLEET_NAME}", token=token)
    
    if result:
        fleet_id = result.get("id")
        fleet_api_key = result.get("api_key")
        print(f"  [OK] Fleet created: {result.get('name')}")
        print(f"  [INFO] Fleet ID: {fleet_id}")
        print(f"  [INFO] API Key: {fleet_api_key}")
        print(f"  [WARN] Save this API key - it will not be shown again!")
    else:
        print("  [ERROR] Fleet creation failed")
        return 1
    
    # Step 4: Create Endpoint
    print_step(4, "Register Endpoint")
    
    result = api_call("POST", "/identity/endpoints", token=token, data={
        "name": "prod-ingress-main",
        "hostname": "api.example.com",
        "port": 443,
        "endpoint_type": "kubernetes",
        "environment": "production",
        "criticality": "critical",
        "fleet_id": fleet_id,
        "k8s_namespace": "default",
        "k8s_secret_name": "api-tls-secret",
        "k8s_ingress_name": "api-ingress",
    })
    
    if result:
        endpoint_id = result.get("id")
        print(f"  [OK] Endpoint registered: {result.get('name')}")
        print(f"  [INFO] Endpoint ID: {endpoint_id}")
    else:
        print("  [ERROR] Endpoint registration failed")
        return 1
    
    # Step 5: Create Policy (200-day preset)
    print_step(5, "Create Certificate Policy")
    
    result = api_call("POST", "/identity/policies", token=token, data={
        "name": "Production TLS Policy",
        "preset_name": "200-day",
    })
    
    if result:
        policy_id = result.get("id")
        print(f"  [OK] Policy created: {result.get('name')}")
        print(f"  [INFO] Policy ID: {policy_id}")
        print(f"  [INFO] Max validity: 200 days")
        print(f"  [INFO] Renewal window: 30 days before expiry")
    else:
        print("  [ERROR] Policy creation failed")
        return 1
    
    # Step 6: Register Agent
    print_step(6, "Register Identity Agent")
    
    result = api_call("POST", "/identity/agent/register", token=token, data={
        "name": "k8s-node-01",
        "hostname": "k8s-node-01.internal",
        "fleet_id": fleet_id,
        "supported_types": ["kubernetes", "nginx", "envoy"],
        "supported_challenges": ["http-01", "dns-01"],
        "version": "1.0.0",
    })
    
    if result:
        agent_id = result.get("agent_id")
        print(f"  [OK] Agent registered: {result.get('name')}")
        print(f"  [INFO] Agent ID: {agent_id}")
    else:
        print("  [ERROR] Agent registration failed")
        return 1
    
    # Step 7: Request Scan
    print_step(7, "Request Certificate Scan")
    
    result = api_call("POST", f"/identity/scan/request?fleet_id={fleet_id}", token=token)
    
    if result:
        print(f"  [OK] Scan requested: {result.get('scan_id')}")
        print(f"  [INFO] Status: {result.get('status')}")
    else:
        print("  [ERROR] Scan request failed")
    
    # Step 8: Schedule Renewal
    print_step(8, "Schedule Renewal Job")
    
    result = api_call("POST", "/identity/renewals/run", token=token, data={
        "endpoint_ids": [endpoint_id],
        "policy_id": policy_id,
    })
    
    if result:
        jobs = result.get("jobs", [])
        if jobs:
            job_id = jobs[0].get("job_id")
            print(f"  [OK] Renewal scheduled: {len(jobs)} job(s)")
            print(f"  [INFO] Job ID: {job_id}")
            print(f"  [INFO] Status: {jobs[0].get('status')}")
        else:
            print("  [WARN] No jobs scheduled")
    else:
        print("  [ERROR] Renewal scheduling failed")
    
    # Step 9: Check Inventory
    print_step(9, "View Inventory")
    
    result = api_call("GET", "/identity/inventory", token=token)
    
    if result:
        print(f"  [INFO] Endpoints: {len(result.get('endpoints', []))}")
        print(f"  [INFO] Certificates: {len(result.get('certificates', []))}")
        print(f"  [INFO] Expiry Summary: {json.dumps(result.get('expiry_summary', {}))}")
    
    # Step 10: Check EKU Conflicts
    print_step(10, "Check EKU Migration")
    
    result = api_call("POST", "/identity/migrations/eku-split", token=token)
    
    if result:
        print(f"  [INFO] Violations found: {result.get('violations_found', 0)}")
        print(f"  [INFO] Chrome deadline: {result.get('chrome_deadline')}")
        if result.get('violations'):
            for v in result['violations']:
                print(f"    - {v.get('subject')}: {v.get('recommendation')}")
    
    # Step 11: View Audit Log
    print_step(11, "View Audit Log")
    
    result = api_call("GET", "/identity/audit?limit=10", token=token)
    
    if result:
        print(f"  [INFO] Recent audit entries:")
        for entry in result[:5]:
            print(f"    [{entry.get('timestamp')}] {entry.get('action')} by {entry.get('actor_type')}")
    
    # Step 12: Verify Audit Chain
    print_step(12, "Verify Audit Chain Integrity")
    
    result = api_call("GET", "/identity/audit/verify", token=token)
    
    if result:
        if result.get("is_valid"):
            print(f"  [OK] Audit chain verified: {result.get('total_entries')} entries")
        else:
            print(f"  [ERROR] Audit chain INVALID: {result.get('error_message')}")
    
    # Summary
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print(f"""
Summary:
  - Tenant: {TENANT_NAME}
  - Fleet: {FLEET_NAME} ({fleet_id})
  - Endpoint: prod-ingress-main ({endpoint_id})
  - Policy: 200-day Public TLS ({policy_id})
  - Agent: k8s-node-01 ({agent_id})

Next Steps:
  1. Run the agent on your K8s nodes:
     python -m tensorguard.identity.agent.main --fleet-id={fleet_id} --api-key=<KEY>
  
  2. The agent will:
     - Scan for certificates in K8s secrets
     - Report findings to control plane
     - Process renewal jobs automatically
     - Deploy renewed certs with rollback support

  3. Monitor via dashboard:
     - Expiry heatmap at /identity/inventory
     - Renewal jobs at /identity/renewals
     - Audit log at /identity/audit
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
