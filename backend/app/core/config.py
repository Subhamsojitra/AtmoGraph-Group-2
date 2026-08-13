"""Centralized application configuration for AtmoGraph.

This module is the single source of truth for runtime settings. Values are
read from environment variables, which take precedence over those defined in a
local ``.env`` file (loaded automatically for local development).

Usage
-----
    from app.core.config import settings

    print(settings.app_name)
    print(settings.neo4j_uri)
    print(settings.neo4j_password.get_secret_value())  # only when truly needed

Notes
-----
* No credentials are hardcoded.
* Neo4j settings are *required*: a missing value raises a clear validation
  error at startup instead of failing later with a cryptic connection error.
* ``NEO4J_PASSWORD`` is parsed as a :class:`pydantic.SecretStr`, so it is never
  rendered in the settings ``repr``/``str`` output.
* Neo4j is *not* connected to here; this module only manages configuration.
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration consumed by the FastAPI application.

    Application-level settings provide sensible, non-secret defaults. Neo4j
    settings are required because AtmoGraph cannot fulfil its core purpose
    without a configured graph database, so missing values must fail fast
    rather than be hidden behind unsafe defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    app_name: str = "AtmoGraph API"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    # Neo4j (used by Module 3 -- database layer)
    # ------------------------------------------------------------------ #
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: SecretStr
    neo4j_database: str


#: Module-level singleton. Import this everywhere instead of instantiating
#: ``Settings`` repeatedly, to avoid creating multiple settings instances.
settings: Settings = Settings()

