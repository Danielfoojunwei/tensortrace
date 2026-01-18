"""
Startup configuration validation.

Centralizes production gate checks for secrets, database configuration,
feature flags, and dependency availability.
"""

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .production_gates import (
    ProductionGateError,
    block_demo_mode,
    is_production,
    require_dependency,
    require_env,
)

logger = logging.getLogger(__name__)

DependencySpec = Tuple[str, Optional[str]]


def _normalize_dependencies(dependencies: Sequence[str] | Sequence[DependencySpec]) -> List[DependencySpec]:
    normalized: List[DependencySpec] = []
    for dep in dependencies:
        if isinstance(dep, tuple):
            normalized.append(dep)
        else:
            normalized.append((dep, None))
    return normalized


def validate_startup_config(
    component: str,
    *,
    require_database: bool = False,
    require_secret_key: bool = False,
    require_key_master: bool = False,
    required_dependencies: Optional[Sequence[str] | Sequence[DependencySpec]] = None,
    feature_flags: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    """
    Validate startup configuration for a component.

    Args:
        component: Name of the component (for logging).
        require_database: Whether DATABASE_URL must be set in production.
        require_secret_key: Whether TG_SECRET_KEY must be set in production.
        require_key_master: Whether TG_KEY_MASTER must be set in production.
        required_dependencies: Dependencies to require in production.
        feature_flags: Optional mapping of feature gates to config:
            {
                "feature_name": {
                    "env_var": "TG_ENABLE_FEATURE",
                    "dependencies": ["module"],
                    "remediation": "Install ..."
                }
            }

    Returns:
        Dict describing validation results.

    Raises:
        ProductionGateError: If any production gate fails.
    """
    results: Dict[str, object] = {
        "component": component,
        "environment": os.getenv("TG_ENVIRONMENT", "development"),
        "is_production": is_production(),
        "gates_checked": [],
    }

    if is_production():
        logger.info("[%s] Running startup validation for production.", component)

    block_demo_mode()
    results["gates_checked"].append("demo_mode_blocked")

    if require_secret_key:
        require_env(
            "TG_SECRET_KEY",
            remediation="Generate a secure key: python -c \"import secrets; print(secrets.token_hex(32))\"",
            min_length=32,
        )
        results["gates_checked"].append("secret_key_set")

    if require_database:
        require_env(
            "DATABASE_URL",
            remediation="Set DATABASE_URL to your PostgreSQL connection string.",
        )
        results["gates_checked"].append("database_url_set")

    if require_key_master:
        require_env(
            "TG_KEY_MASTER",
            remediation="Generate a 32-byte hex key: python -c \"import os; print(os.urandom(32).hex())\"",
            min_length=64,
        )
        results["gates_checked"].append("key_master_set")

    if required_dependencies:
        for dep_name, remediation in _normalize_dependencies(required_dependencies):
            require_dependency(
                dep_name,
                remediation=remediation or f"Install dependency: pip install {dep_name}",
            )
            results["gates_checked"].append(f"dependency:{dep_name}")

    if feature_flags:
        for feature_name, config in feature_flags.items():
            env_var = str(config.get("env_var", ""))
            if not env_var:
                raise ValueError(f"Feature flag '{feature_name}' missing env_var")
            enabled = os.getenv(env_var, "false").lower() == "true"
            if not enabled:
                continue
            deps = config.get("dependencies", [])
            remediation = config.get("remediation")
            for dep_name, dep_remediation in _normalize_dependencies(deps):
                require_dependency(
                    dep_name,
                    remediation=dep_remediation or remediation or f"Install dependency: pip install {dep_name}",
                )
                results["gates_checked"].append(f"feature:{feature_name}:{dep_name}")

    return results
