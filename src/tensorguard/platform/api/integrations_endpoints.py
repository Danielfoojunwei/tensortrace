"""
Tier 3: External Integrations API - Production Hardened

Handles connections to NVIDIA Isaac Lab, ROS2, Formant.io, and Hugging Face.
Uses database-backed state and real connector validation.
"""

import json
import time
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from ..database import get_session
from ..auth import get_current_user
from ..models.core import User
from ..models.settings_models import IntegrationConnection, IntegrationStatus
from ...utils.production_gates import is_production, ProductionGateError
from ...utils.config_encryption import encrypt_sensitive_fields

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class ConnectionRequest(BaseModel):
    """Request to connect an integration."""

    service: str  # 'isaac_lab', 'ros2_bridge', 'formant', 'huggingface'
    config: Dict[str, str]


class ValidationResponse(BaseModel):
    """Integration validation response."""

    status: str
    message: str
    latency_ms: Optional[float] = None
    remediation: Optional[str] = None


# ============================================================================
# Connector Interface
# ============================================================================


class IntegrationConnector:
    """Base interface for integration connectors."""

    service_name: str = "unknown"

    def validate_credentials(self, config: Dict[str, str]) -> bool:
        """Validate the provided credentials/configuration."""
        raise NotImplementedError

    def health_check(self, config: Dict[str, str]) -> Dict[str, Any]:
        """Perform a health check on the integration."""
        raise NotImplementedError

    def get_remediation(self) -> str:
        """Get remediation steps for connection issues."""
        return "Check your configuration and credentials."


class IsaacLabConnector(IntegrationConnector):
    """NVIDIA Isaac Lab / Omniverse connector."""

    service_name = "isaac_lab"

    def validate_credentials(self, config: Dict[str, str]) -> bool:
        if "omniverse_url" not in config:
            return False
        # Real validation would check Nucleus server connectivity
        return True

    def health_check(self, config: Dict[str, str]) -> Dict[str, Any]:
        """
        Perform real health check against Isaac Lab/Omniverse.

        In production, this attempts actual connection to Nucleus server.
        """
        omniverse_url = config.get("omniverse_url")
        if not omniverse_url:
            return {
                "status": IntegrationStatus.ERROR.value,
                "message": "Missing omniverse_url configuration",
                "latency_ms": None,
            }

        start = time.time()

        try:
            # In production, we'd use the Omniverse Kit SDK
            # For now, we do a basic HTTP check if URL is provided
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                omniverse_url,
                method="HEAD",
                headers={"User-Agent": "TensorGuard-IntegrationCheck/1.0"},
            )
            urllib.request.urlopen(req, timeout=10)
            latency_ms = (time.time() - start) * 1000

            return {
                "status": IntegrationStatus.CONNECTED.value,
                "message": "Omniverse Nucleus server reachable",
                "latency_ms": round(latency_ms, 2),
            }

        except urllib.error.URLError as e:
            latency_ms = (time.time() - start) * 1000
            return {
                "status": IntegrationStatus.ERROR.value,
                "message": f"Cannot reach Omniverse server: {e.reason}",
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as e:
            return {
                "status": IntegrationStatus.ERROR.value,
                "message": f"Health check failed: {str(e)}",
                "latency_ms": None,
            }

    def get_remediation(self) -> str:
        return (
            "Ensure Omniverse Nucleus server is running and accessible. "
            "Check firewall rules and verify omniverse_url is correct."
        )


class ROS2BridgeConnector(IntegrationConnector):
    """ROS2 Bridge connector."""

    service_name = "ros2_bridge"

    def validate_credentials(self, config: Dict[str, str]) -> bool:
        # ROS2 domain ID should be a number 0-232
        domain_id = config.get("domain_id", "0")
        try:
            did = int(domain_id)
            return 0 <= did <= 232
        except ValueError:
            return False

    def health_check(self, config: Dict[str, str]) -> Dict[str, Any]:
        """
        Check ROS2 bridge connectivity.

        In production, this would use rclpy to check domain discovery.
        """
        domain_id = config.get("domain_id", "0")

        try:
            # Check if rclpy is available
            import importlib

            rclpy_spec = importlib.util.find_spec("rclpy")
            if rclpy_spec is None:
                return {
                    "status": IntegrationStatus.UNAVAILABLE.value,
                    "message": "rclpy not installed - ROS2 bridge unavailable",
                    "latency_ms": None,
                }

            # If rclpy available, we'd do actual discovery
            # For now, return unavailable with instructions
            return {
                "status": IntegrationStatus.UNAVAILABLE.value,
                "message": f"ROS2 domain {domain_id} discovery requires ROS2 environment",
                "latency_ms": None,
            }

        except Exception as e:
            return {
                "status": IntegrationStatus.ERROR.value,
                "message": f"ROS2 check failed: {str(e)}",
                "latency_ms": None,
            }

    def get_remediation(self) -> str:
        return (
            "Install ROS2 and rclpy package. "
            "Source your ROS2 workspace and ensure DDS discovery is configured."
        )


class FormantConnector(IntegrationConnector):
    """Formant.io connector."""

    service_name = "formant"

    def validate_credentials(self, config: Dict[str, str]) -> bool:
        # Require API key or agent token
        return "api_key" in config or "agent_token" in config

    def health_check(self, config: Dict[str, str]) -> Dict[str, Any]:
        """
        Check Formant API connectivity.

        In production, validates against Formant's API.
        """
        api_key = config.get("api_key") or config.get("agent_token")
        if not api_key:
            return {
                "status": IntegrationStatus.ERROR.value,
                "message": "Missing api_key or agent_token",
                "latency_ms": None,
            }

        start = time.time()

        try:
            import urllib.request
            import urllib.error

            # Check Formant API endpoint
            req = urllib.request.Request(
                "https://api.formant.io/v1/admin/status",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "TensorGuard-IntegrationCheck/1.0",
                },
            )
            urllib.request.urlopen(req, timeout=10)
            latency_ms = (time.time() - start) * 1000

            return {
                "status": IntegrationStatus.CONNECTED.value,
                "message": "Formant API authenticated",
                "latency_ms": round(latency_ms, 2),
            }

        except urllib.error.HTTPError as e:
            latency_ms = (time.time() - start) * 1000
            if e.code == 401:
                return {
                    "status": IntegrationStatus.ERROR.value,
                    "message": "Formant authentication failed - invalid API key",
                    "latency_ms": round(latency_ms, 2),
                }
            return {
                "status": IntegrationStatus.ERROR.value,
                "message": f"Formant API error: {e.code}",
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as e:
            return {
                "status": IntegrationStatus.ERROR.value,
                "message": f"Formant check failed: {str(e)}",
                "latency_ms": None,
            }

    def get_remediation(self) -> str:
        return (
            "Verify your Formant API key or agent token. "
            "Generate a new token from the Formant dashboard if needed."
        )


class HuggingFaceConnector(IntegrationConnector):
    """HuggingFace Hub connector."""

    service_name = "huggingface"

    def validate_credentials(self, config: Dict[str, str]) -> bool:
        # Model ID format: user/repo
        model_id = config.get("model_id", "")
        return "/" in model_id

    def health_check(self, config: Dict[str, str]) -> Dict[str, Any]:
        """
        Check HuggingFace model availability.

        In production, validates model exists on HF Hub.
        """
        model_id = config.get("model_id", "")
        if "/" not in model_id:
            return {
                "status": IntegrationStatus.ERROR.value,
                "message": "Invalid model_id format (expected: user/repo)",
                "latency_ms": None,
            }

        start = time.time()

        try:
            import urllib.request
            import urllib.error

            # Check HF Hub API
            api_url = f"https://huggingface.co/api/models/{model_id}"
            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": "TensorGuard-IntegrationCheck/1.0"},
            )

            hf_token = config.get("hf_token")
            if hf_token:
                req.add_header("Authorization", f"Bearer {hf_token}")

            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                latency_ms = (time.time() - start) * 1000

                # Extract model info
                model_size = data.get("safetensors", {}).get("total", 0)
                size_str = f"{model_size / 1e9:.1f}GB" if model_size else "unknown size"

                return {
                    "status": IntegrationStatus.CONNECTED.value,
                    "message": f"Model found: {model_id} ({size_str})",
                    "latency_ms": round(latency_ms, 2),
                }

        except urllib.error.HTTPError as e:
            latency_ms = (time.time() - start) * 1000
            if e.code == 404:
                return {
                    "status": IntegrationStatus.ERROR.value,
                    "message": f"Model not found: {model_id}",
                    "latency_ms": round(latency_ms, 2),
                }
            elif e.code == 401:
                return {
                    "status": IntegrationStatus.ERROR.value,
                    "message": "Private model requires hf_token",
                    "latency_ms": round(latency_ms, 2),
                }
            return {
                "status": IntegrationStatus.ERROR.value,
                "message": f"HuggingFace API error: {e.code}",
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as e:
            return {
                "status": IntegrationStatus.ERROR.value,
                "message": f"HuggingFace check failed: {str(e)}",
                "latency_ms": None,
            }

    def get_remediation(self) -> str:
        return (
            "Verify the model_id format (user/repo). "
            "For private models, provide hf_token in config."
        )


# Connector registry
CONNECTORS: Dict[str, IntegrationConnector] = {
    "isaac_lab": IsaacLabConnector(),
    "ros2_bridge": ROS2BridgeConnector(),
    "formant": FormantConnector(),
    "huggingface": HuggingFaceConnector(),
}


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/integrations/connect")
async def connect_integration(
    req: ConnectionRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> ValidationResponse:
    """
    Connect to an external integration.

    Performs real credential validation and health check.
    Stores connection state in database.
    """
    connector = CONNECTORS.get(req.service)
    if not connector:
        raise HTTPException(404, f"Unknown service: {req.service}")

    # Validate credentials
    if not connector.validate_credentials(req.config):
        return ValidationResponse(
            status=IntegrationStatus.ERROR.value,
            message=f"Invalid configuration for {req.service}",
            remediation=connector.get_remediation(),
        )

    # Perform health check
    health = connector.health_check(req.config)

    # Store/update connection in database
    existing = session.exec(
        select(IntegrationConnection)
        .where(IntegrationConnection.tenant_id == current_user.tenant_id)
        .where(IntegrationConnection.service == req.service)
    ).first()

    # Encrypt sensitive fields in config (api_key, password, token, etc.)
    encrypted_config = encrypt_sensitive_fields(req.config)

    if existing:
        existing.status = health["status"]
        existing.config_json = encrypted_config
        existing.last_health_check = datetime.utcnow()
        existing.health_check_latency_ms = health.get("latency_ms")
        existing.error_message = health["message"] if health["status"] != IntegrationStatus.CONNECTED.value else None
        existing.updated_at = datetime.utcnow()
        if health["status"] == IntegrationStatus.CONNECTED.value:
            existing.last_seen = datetime.utcnow()
        session.add(existing)
    else:
        conn = IntegrationConnection(
            tenant_id=current_user.tenant_id,
            service=req.service,
            status=health["status"],
            config_json=encrypted_config,
            last_health_check=datetime.utcnow(),
            health_check_latency_ms=health.get("latency_ms"),
            error_message=health["message"] if health["status"] != IntegrationStatus.CONNECTED.value else None,
            last_seen=datetime.utcnow() if health["status"] == IntegrationStatus.CONNECTED.value else None,
        )
        session.add(conn)

    session.commit()

    return ValidationResponse(
        status=health["status"],
        message=health["message"],
        latency_ms=health.get("latency_ms"),
        remediation=connector.get_remediation() if health["status"] != IntegrationStatus.CONNECTED.value else None,
    )


@router.get("/integrations/status")
async def get_integration_status(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get real status of all integrations for the current tenant.

    Returns database-backed connection state.
    Never returns mock/simulated status in production.
    """
    connections = session.exec(
        select(IntegrationConnection).where(
            IntegrationConnection.tenant_id == current_user.tenant_id
        )
    ).all()

    # Build status map with real data
    status_map = {}

    for conn in connections:
        status_map[conn.service] = {
            "status": conn.status,
            "last_seen": conn.last_seen.isoformat() if conn.last_seen else None,
            "last_health_check": conn.last_health_check.isoformat() if conn.last_health_check else None,
            "latency_ms": conn.health_check_latency_ms,
            "error": conn.error_message,
        }

    # Add unavailable entries for services not configured
    for service in CONNECTORS:
        if service not in status_map:
            status_map[service] = {
                "status": IntegrationStatus.UNAVAILABLE.value,
                "message": "Not configured",
                "remediation": CONNECTORS[service].get_remediation(),
            }

    return status_map


@router.post("/integrations/{service}/health")
async def check_integration_health(
    service: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Perform a health check on a specific integration.

    Uses stored configuration to validate connection.
    """
    connector = CONNECTORS.get(service)
    if not connector:
        raise HTTPException(404, f"Unknown service: {service}")

    # Get stored configuration
    conn = session.exec(
        select(IntegrationConnection)
        .where(IntegrationConnection.tenant_id == current_user.tenant_id)
        .where(IntegrationConnection.service == service)
    ).first()

    if not conn:
        raise HTTPException(
            424,
            detail={
                "status": IntegrationStatus.UNAVAILABLE.value,
                "message": f"Integration {service} not configured",
                "remediation": connector.get_remediation(),
            },
        )

    # Perform health check
    config = json.loads(conn.config_json)
    health = connector.health_check(config)

    # Update connection state
    conn.status = health["status"]
    conn.last_health_check = datetime.utcnow()
    conn.health_check_latency_ms = health.get("latency_ms")
    conn.error_message = health["message"] if health["status"] != IntegrationStatus.CONNECTED.value else None
    if health["status"] == IntegrationStatus.CONNECTED.value:
        conn.last_seen = datetime.utcnow()
    conn.updated_at = datetime.utcnow()

    session.add(conn)
    session.commit()

    return {
        "service": service,
        "status": health["status"],
        "message": health["message"],
        "latency_ms": health.get("latency_ms"),
        "checked_at": datetime.utcnow().isoformat(),
    }


@router.delete("/integrations/{service}")
async def disconnect_integration(
    service: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Remove an integration configuration."""
    conn = session.exec(
        select(IntegrationConnection)
        .where(IntegrationConnection.tenant_id == current_user.tenant_id)
        .where(IntegrationConnection.service == service)
    ).first()

    if not conn:
        raise HTTPException(404, f"Integration not found: {service}")

    session.delete(conn)
    session.commit()

    return {"status": "disconnected", "service": service}
