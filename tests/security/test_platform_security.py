import unittest
import os
import shutil
from fastapi.testclient import TestClient
from tensorguard.platform.main import app

class TestPlatformSecurity(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_upload_path_traversal_sanitization(self):
        """Test that uploaded filenames with ../ are sanitized."""
        import secrets
        filename = "../../../evil_script.py"
        # Create a dummy TGSP file
        from unittest.mock import patch
        
        mock_manifest = {
            "package_id": f"evil-{secrets.token_hex(4)}",
            "author_id": "malicious",
            "created_at": 1735732800.0,
            "policy_id": "none",
            "policy_version": 1,
            "content_index": [],
            "compat_base_model_id": []
        }
        
        with patch("tensorguard.platform.api.community_tgsp.read_tgsp_header") as mock_read:
            mock_read.return_value = {
                "version": "0.2",
                "header": {"hashes": {"manifest": "hash123"}},
                "manifest": mock_manifest,
                "recipients": []
            }
            
            response = self.client.post(
                "/api/community/tgsp/upload",
                files={"file": (filename, b"fake_tgsp_binary", "application/octet-stream")}
            )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify the saved filename is just 'evil_script.py'
        self.assertEqual(data["filename"], "evil_script.py")
        
        # Verify it's in the designated storage dir, not root
        storage_path = data["storage_path"]
        self.assertIn("storage", storage_path)
        self.assertNotIn("..", storage_path)

if __name__ == "__main__":
    unittest.main()
