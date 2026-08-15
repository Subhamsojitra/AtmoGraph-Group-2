"""Tests for centralized configuration (Module 2).

These tests exercise the Pydantic-Settings based configuration in
``app.core.config`` without requiring a running Neo4j server.  Environment
variables are set via ``pytest``'s ``monkeypatch`` fixture so that no real
credentials are ever needed and no secrets are printed.
"""

import importlib
import os
import shutil

import pytest
from pydantic import SecretStr, ValidationError


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

# Fake (non-real) Neo4j credentials used solely for exercising configuration
# loading.  None of these values correspond to a real database.
_FAKE_NEO4J_ENV = {
    "NEO4J_URI": "neo4j://test-host:7687",
    "NEO4J_USERNAME": "test_user",
    "NEO4J_PASSWORD": "test_super_secret_123",
    "NEO4J_DATABASE": "test_db",
}


def _set_neo4j_env(monkeypatch):
    """Populate the required Neo4j environment variables via monkeypatch."""
    for key, value in _FAKE_NEO4J_ENV.items():
        monkeypatch.setenv(key, value)


def _load_config(monkeypatch):
    """Set required Neo4j env vars and reload the config module.

    The config module creates a module-level ``Settings()`` singleton at
    import time.  Because Neo4j fields are required, the environment must
    contain valid values before the module is (re)imported.  This helper
    ensures that precondition and returns the freshly reloaded module.
    """
    _set_neo4j_env(monkeypatch)
    from app.core import config
    importlib.reload(config)
    return config


# ---------------------------------------------------------------------------
# 1. Default application settings
# ---------------------------------------------------------------------------

def test_default_application_settings(monkeypatch):
    """Application-level settings fall back to documented defaults when the
    corresponding environment variables are not set."""
    _set_neo4j_env(monkeypatch)
    # Ensure app-level env overrides are absent so defaults are exercised.
    monkeypatch.delenv("APP_NAME", raising=False)
    monkeypatch.delenv("APP_VERSION", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    from app.core import config
    importlib.reload(config)

    assert config.settings.app_name == "AtmoGraph API"
    assert config.settings.app_version == "0.1.0"
    assert config.settings.log_level == "INFO"


# ---------------------------------------------------------------------------
# 2. Environment variable overrides using pytest monkeypatch
# ---------------------------------------------------------------------------

def test_environment_variable_overrides(monkeypatch):
    """Environment variables override the default application settings."""
    _set_neo4j_env(monkeypatch)
    monkeypatch.setenv("APP_NAME", "Custom App Name")
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    from app.core import config
    importlib.reload(config)

    assert config.settings.app_name == "Custom App Name"
    assert config.settings.app_version == "9.9.9"
    assert config.settings.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# 3. Neo4j configuration loading
# ---------------------------------------------------------------------------

def test_neo4j_configuration_loading(monkeypatch):
    """Neo4j configuration fields are populated from environment variables."""
    config = _load_config(monkeypatch)

    assert config.settings.neo4j_uri == "neo4j://test-host:7687"
    assert config.settings.neo4j_username == "test_user"
    assert config.settings.neo4j_database == "test_db"
    assert config.settings.neo4j_password.get_secret_value() == "test_super_secret_123"


# ---------------------------------------------------------------------------
# 4. NEO4J_PASSWORD uses SecretStr and does not expose the actual password
# ---------------------------------------------------------------------------

def test_neo4j_password_uses_secret_str(monkeypatch):
    """NEO4J_PASSWORD is parsed as a SecretStr and never exposes the raw
    password in its normal ``str`` / ``repr`` representation."""
    config = _load_config(monkeypatch)

    password = config.settings.neo4j_password

    # Type check — must be a SecretStr, not a plain str.
    assert isinstance(password, SecretStr)

    # The raw password must not leak into any human-readable representation.
    raw = "test_super_secret_123"
    assert raw not in repr(password)
    assert raw not in str(password)
    assert raw not in repr(config.settings)
    assert raw not in str(config.settings)

    # Only an explicit ``get_secret_value()`` call reveals the password.
    assert password.get_secret_value() == raw


# ---------------------------------------------------------------------------
# 5. Missing required Neo4j configuration raises a Pydantic validation error
# ---------------------------------------------------------------------------

def test_missing_neo4j_config_raises_validation_error(monkeypatch):
    """When required Neo4j environment variables are absent, instantiating
    ``Settings`` raises a Pydantic ``ValidationError``."""
    # Set env vars first so the module can be imported (singleton is created).
    _set_neo4j_env(monkeypatch)

    # Remove every required Neo4j env var to simulate a misconfigured
    # environment.
    for key in _FAKE_NEO4J_ENV:
        monkeypatch.delenv(key, raising=False)

    # Reload the config module to get a fresh Settings instance
    # that will see the missing environment variables
    from app.core import config
    importlib.reload(config)
    from app.core.config import Settings

    # Create a new Settings instance that ignores the .env file
    # to properly test missing configuration
    with pytest.raises(ValidationError):
        Settings(_env_file=None)