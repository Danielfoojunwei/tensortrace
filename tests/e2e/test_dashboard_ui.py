
import pytest
import requests

class TestDashboardE2E:
    """End-to-End tests for Dashboard UI and API."""
    
    def test_dashboard_home_loads(self, api_server):
        """Verify dashboard homepage loads (StaticFiles)."""
        response = requests.get(f"{api_server}/")
        assert response.status_code == 200
        assert "TensorGuard Enterprise PLM" in response.text
        assert '<div id="app">' in response.text

    def test_static_assets(self, api_server):
        """Verify CSS and JS load."""
        files = ["styles.css", "app.js"]
        for f in files:
            resp = requests.get(f"{api_server}/{f}")
            assert resp.status_code == 200, f"Failed to load {f}"

    def test_enablement_stats_api(self, api_server):
        """Verify stats endpoint used by dashboard."""
        resp = requests.get(f"{api_server}/api/v1/enablement/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_jobs" in data
        assert "pending_jobs" in data
    
    def test_api_docs_accessible(self, api_server):
        """Verify Swagger UI is up."""
        resp = requests.get(f"{api_server}/docs")
        assert resp.status_code == 200
        assert "Swagger UI" in resp.text or "OpenAPI" in resp.text

