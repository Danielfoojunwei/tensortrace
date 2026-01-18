import os
import pytest

from tensorguard.utils.production_gates import ProductionGateError, is_production
from tensorguard.utils.startup_validation import validate_startup_config


def test_startup_validation_requires_secret_key_in_production(monkeypatch):
    monkeypatch.setenv("TG_ENVIRONMENT", "production")
    monkeypatch.delenv("TG_SECRET_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("TG_KEY_MASTER", "a" * 64)
    is_production.cache_clear()

    with pytest.raises(ProductionGateError):
        validate_startup_config("platform", require_secret_key=True, require_database=True, require_key_master=True)


def test_startup_validation_allows_dev_without_secrets(monkeypatch):
    monkeypatch.setenv("TG_ENVIRONMENT", "development")
    monkeypatch.delenv("TG_SECRET_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TG_KEY_MASTER", raising=False)
    is_production.cache_clear()

    validate_startup_config("platform", require_secret_key=True, require_database=True, require_key_master=True)
