import requests
import sys

BASE = "http://127.0.0.1:8000/api/v1"

def test():
    print("Testing init...")
    try:
        # 1. Init Tenant
        res = requests.post(f"{BASE}/onboarding/init", params={
            "name": "TestCorp",
            "admin_email": "admin@test.com",
            "admin_pass": "password123"
        })
        if res.status_code == 200:
            print("Init success:", res.json())
        else:
            print("Init failed (might be already present):", res.text)
            
        # 2. Login
        print("Testing login...")
        res = requests.post(f"{BASE}/auth/token", json={
            "username": "admin@test.com",
            "password": "password123"
        })
        if res.status_code == 200:
            token = res.json()["access_token"]
            print("Login success, token obtained")
            
            # 3. Get Fleets
            headers = {"Authorization": f"Bearer {token}"}
            res = requests.get(f"{BASE}/fleets", headers=headers)
            print("Fleets:", res.json())
            
            # 4. Create Fleet
            res = requests.post(f"{BASE}/fleets", params={"name": "RobotFleet1"}, headers=headers)
            print("Created Fleet:", res.json())
            
        else:
            print("Login failed:", res.text)
            sys.exit(1)
            
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    test()
