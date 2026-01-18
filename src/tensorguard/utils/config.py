"""
TensorGuard Configuration Module

Provides centralized configuration management with:
- Environment variable loading (TG_ prefix preferred, TENSORGUARD_ supported)
- Type validation via Pydantic
- Secure defaults for production
- Development overrides via .env file

Environment Variable Naming Convention:
- All variables use TG_ prefix (e.g., TG_ENVIRONMENT, TG_SECRET_KEY)
- Legacy TENSORGUARD_ prefix is supported for backward compatibility
- Database uses standard DATABASE_URL for ORM compatibility
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional, List
import os


class TensorGuardSettings(BaseSettings):
    """
    TensorGuard System settings.

    Loads from environment variables with TG_ prefix (preferred).
    Also supports TENSORGUARD_ prefix for backward compatibility.

    Usage:
        from tensorguard.utils.config import settings

        if settings.ENVIRONMENT == "production":
            # Use strict security settings
    """
    model_config = SettingsConfigDict(
        env_prefix='TG_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # ==========================================================================
    # GENERAL
    # ==========================================================================
    ENVIRONMENT: str = Field(default="development", description="Runtime environment: development, staging, production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")

    # ==========================================================================
    # SECURITY
    # ==========================================================================
    SECRET_KEY: Optional[str] = Field(default=None, description="JWT signing key (required in production)")
    DEMO_MODE: bool = Field(default=False, description="Enable demo mode (NEVER in production)")
    PQC_STRICT: bool = Field(default=True, description="Require real PQC libs (auto-enabled in production)")
    COMMUNITY_MODE: str = Field(default="true", description="Community edition mode: true, permissive")

    # ==========================================================================
    # AUTHENTICATION
    # ==========================================================================
    TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token lifetime in minutes")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token lifetime in days")
    TOKEN_ISSUER: str = Field(default="tensorguard-platform", description="JWT issuer claim")
    TOKEN_AUDIENCE: str = Field(default="tensorguard-api", description="JWT audience claim")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    MIN_PASSWORD_LENGTH: int = Field(default=12, description="Minimum password length")
    REQUIRE_PASSWORD_COMPLEXITY: bool = Field(default=True, description="Require mixed case, digits, special chars")

    # ==========================================================================
    # CORS
    # ==========================================================================
    ALLOWED_ORIGINS: str = Field(default="", description="Comma-separated allowed origins (empty = restrictive default)")
    ALLOW_CREDENTIALS: bool = Field(default=False, description="Allow credentials in CORS (requires explicit origins)")

    # ==========================================================================
    # DATABASE
    # ==========================================================================
    DATABASE_URL: str = Field(default="sqlite:///./tensorguard.db", description="Database connection URL")

    # ==========================================================================
    # CRYPTO PARAMETERS
    # ==========================================================================
    SECURITY_LEVEL: int = Field(default=128, description="Security level in bits (128, 192, 256)")
    MAX_KEY_USES: int = Field(default=1000, description="Maximum uses before key rotation")
    LATTICE_DIMENSION: int = Field(default=1024, description="LWE lattice dimension")
    PLAINTEXT_MODULUS: int = Field(default=65536, description="Plaintext modulus for HE")

    # ==========================================================================
    # PRIVACY PIPELINE (DP)
    # ==========================================================================
    DP_EPSILON: float = Field(default=1.0, description="Total DP epsilon budget")
    DP_DELTA: float = Field(default=1e-5, description="DP delta parameter")
    DEFAULT_SPARSITY: float = Field(default=0.01, description="Default gradient sparsity ratio")
    DEFAULT_COMPRESSION: int = Field(default=32, description="Default compression ratio")
    MAX_GRADIENT_NORM: float = Field(default=1.0, description="Gradient clipping norm")

    # ==========================================================================
    # NETWORKING
    # ==========================================================================
    CLOUD_ENDPOINT: str = Field(default="https://api.tensor-crate.ai", description="Cloud API endpoint")
    CONTROL_PLANE_URL: str = Field(default="http://localhost:8000", description="Local or external control plane URL")
    DEFAULT_PORT: int = Field(default=8080, description="Default server port")
    DASHBOARD_PORT: int = Field(default=8000, description="Dashboard/platform port")
    MIN_CLIENTS: int = Field(default=2, description="Minimum clients for FL aggregation")

    # ==========================================================================
    # OBSERVABILITY
    # ==========================================================================
    ENABLE_OTEL: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    OTEL_ENDPOINT: str = Field(default="http://localhost:4317", description="OTEL collector endpoint")
    ENABLE_PROMETHEUS: bool = Field(default=False, description="Enable Prometheus metrics")
    PROMETHEUS_PORT: int = Field(default=9090, description="Prometheus metrics port")
    ENABLE_SECURITY_HEADERS: bool = Field(default=True, description="Add security headers to responses")

    # ==========================================================================
    # PATHS
    # ==========================================================================
    KEY_PATH: str = Field(default="keys/enterprise_key.npy", description="Default key storage path")
    ARTIFACTS_PATH: str = Field(default="artifacts/", description="Artifacts storage path")

    # ==========================================================================
    # RATE LIMITING
    # ==========================================================================
    ENABLE_RATE_LIMITING: bool = Field(default=False, description="Enable rate limiting (requires Redis)")
    MAX_LOGIN_ATTEMPTS: int = Field(default=5, description="Max login attempts before lockout")
    LOCKOUT_DURATION_MINUTES: int = Field(default=15, description="Lockout duration after max attempts")

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    def validate_production_config(self) -> List[str]:
        """
        Validate configuration for production readiness.

        Returns:
            List of configuration warnings/errors
        """
        issues = []

        if self.is_production():
            if not self.SECRET_KEY:
                issues.append("CRITICAL: TG_SECRET_KEY not set")
            if self.DEMO_MODE:
                issues.append("CRITICAL: TG_DEMO_MODE=true in production")
            if not self.PQC_STRICT:
                issues.append("WARNING: TG_PQC_STRICT=false in production")
            if "*" in self.ALLOWED_ORIGINS:
                issues.append("WARNING: Wildcard CORS origin in production")
            if self.ALLOW_CREDENTIALS and not self.ALLOWED_ORIGINS:
                issues.append("WARNING: Credentials enabled without explicit origins")

        return issues


# Global settings instance
settings = TensorGuardSettings()

# Validate on import if in production
if settings.is_production():
    _issues = settings.validate_production_config()
    if _issues:
        import logging
        _logger = logging.getLogger(__name__)
        for issue in _issues:
            if issue.startswith("CRITICAL"):
                _logger.critical(issue)
            else:
                _logger.warning(issue)
