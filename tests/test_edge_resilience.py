import unittest
from unittest.mock import patch, MagicMock
import requests
from tensorguard.edge.tgsp_client import TGSPEdgeClient, TGSPClientError

class TestEdgeResilience(unittest.TestCase):
    def setUp(self):
        self.client = TGSPEdgeClient("http://mock-server")

    def test_retry_configuration(self):
        """Verify that the client is configured with a retry strategy."""
        adapter = self.client.session.get_adapter("http://")
        self.assertIsNotNone(adapter.max_retries)
        self.assertEqual(adapter.max_retries.total, 3)
        self.assertIn(500, adapter.max_retries.status_forcelist)
        self.assertIn(502, adapter.max_retries.status_forcelist)

    @patch('requests.Session.get')
    def test_timeout_handling(self, mock_get):
        """Test that the client raises TGSPClientError on timeout."""
        mock_get.side_effect = requests.exceptions.Timeout("Timeout occurred")
        
        with self.assertRaises(TGSPClientError) as cm:
            self.client.get_latest_package("fleet123")
        
        self.assertIn("Platform communication failure", str(cm.exception))

    @patch('requests.Session.get')
    def test_success_flow(self, mock_get):
        """Test basic success path."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_resp
        
        result = self.client.get_latest_package("fleet123")
        self.assertEqual(result["status"], "ok")

if __name__ == "__main__":
    unittest.main()
