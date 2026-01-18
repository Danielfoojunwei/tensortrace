
import pytest
import subprocess
import time
import requests
import os
import sys
import signal

# Define server configuration
HOST = "127.0.0.1"
PORT = 8001
BASE_URL = f"http://{HOST}:{PORT}"

@pytest.fixture(scope="session")
def api_server():
    """Start the FastAPI server as a subprocess."""
    # Use a separate test DB for E2E
    env = os.environ.copy()
    env["DATABASE_URL"] = "sqlite:///./test_e2e.db"
    env["PYTHONPATH"] = os.path.abspath("src")
    
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "tensorguard.platform.main:app", 
        "--host", HOST, 
        "--port", str(PORT)
    ]
    
    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to come up
    limit = 10 # Seconds
    start_time = time.time()
    active = False
    
    while time.time() - start_time < limit:
        try:
            requests.get(f"{BASE_URL}/docs", timeout=1)
            active = True
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
            
    if not active:
        proc.kill()
        out, err = proc.communicate()
        pytest.fail(f"Server failed to start within {limit} seconds.\nSTDOUT: {out.decode()}\nSTDERR: {err.decode()}")
        
    yield BASE_URL
    
    # Cleanup
    print("Stopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        
    # Remove test db
    if os.path.exists("test_e2e.db"):
        os.remove("test_e2e.db")
