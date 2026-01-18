import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from tensorguard.utils.files import atomic_write, sanitize_path
from tensorguard.utils.http import get_standard_client
from tensorguard.utils.exceptions import CommunicationError

def test_atomic_write(tmp_path):
    target = tmp_path / "test.json"
    data = {"key": "value"}
    atomic_write(target, json.dumps(data))
    
    assert target.exists()
    assert json.loads(target.read_text()) == data

def test_sanitize_path():
    assert sanitize_path("../../etc/passwd") == Path("passwd")
    assert sanitize_path("safe.json", "/tmp") == Path("/tmp/safe.json")

@patch("requests.Session.request")
def test_standard_http_client(mock_request):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"status": "ok"}'
    mock_response.json.return_value = {"status": "ok"}
    mock_request.return_value = mock_response
    
    client = get_standard_client("http://localhost:8080", "test-key")
    res = client.request("GET", "api/status")
    
    assert res == {"status": "ok"}
    mock_request.assert_called_once()
    assert mock_request.call_args[0][1] == "http://localhost:8080/api/status"

@patch("requests.Session.request")
def test_standard_http_client_error(mock_request):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    import requests
    mock_request.side_effect = requests.exceptions.HTTPError(response=mock_response)
    
    client = get_standard_client("http://localhost:8080")
    with pytest.raises(CommunicationError):
        client.request("GET", "api/fail")
