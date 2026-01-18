import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict, Any

from .logging import get_logger
from .exceptions import CommunicationError

logger = get_logger(__name__)

# Connection pool configuration
DEFAULT_POOL_CONNECTIONS = 10  # Number of connection pools
DEFAULT_POOL_MAXSIZE = 20      # Max connections per pool
DEFAULT_POOL_BLOCK = False     # Don't block when pool exhausted


class StandardClient:
    """
    Standard HTTP Client for TensorGuard.
    Enforces timeouts, retries, connection pooling, and standard headers.
    """
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        pool_connections: int = DEFAULT_POOL_CONNECTIONS,
        pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

        # Configure Retries with exponential backoff
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"],
        )

        # Configure HTTPAdapter with connection pooling for better performance
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            pool_block=DEFAULT_POOL_BLOCK,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Standard Headers with gzip support
        self.session.headers.update({
            "User-Agent": "TensorGuard-Client/2.1.0",
            "X-TG-Client-Version": "2.1.0",
            "Accept-Encoding": "gzip, deflate",
        })
        if api_key:
            self.session.headers.update({"X-TG-Fleet-API-Key": api_key})

    def request(self, method: str, path: str, timeout: int = 15, **kwargs) -> Dict[str, Any]:
        """Performs a request and handles common errors."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        
        try:
            response = self.session.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            raise CommunicationError(f"API Error: {e.response.status_code}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Network Error: {e}")
            raise CommunicationError("Failed to connect to Control Plane") from e

def get_standard_client(base_url: str, api_key: Optional[str] = None) -> StandardClient:
    return StandardClient(base_url, api_key)
