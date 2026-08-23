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

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Environment file location
# --------------------------------------------------------------------------- #
# The ``.env`` file lives at the repository root (see ``.env.example``).
# Its path is derived here as an *absolute* path from this module's location
# on disk instead of using a relative ``"../.env"``.  Pydantic-Settings
# resolves a relative ``env_file`` against the process working directory, so
# the application failed with "Field required" for every Neo4j setting anytime
# it was launched from anywhere other than ``backend/`` (e.g. the repository
# root).  Deriving the path from ``__file__`` makes configuration loading
# independent of the current working directory.
_BACKEND_DIR = Path(__file__).resolve().parents[2]  # .../backend
_PROJECT_ROOT = _BACKEND_DIR.parent                # .../AtmoGraph-Group-2
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Runtime configuration consumed by the FastAPI application.

    Application-level settings provide sensible, non-secret defaults. Neo4j
    settings are required because AtmoGraph cannot fulfil its core purpose
    without a configured graph database, so missing values must fail fast
    rather than be hidden behind unsafe defaults.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
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

    # ------------------------------------------------------------------ #
    # NLP (Module 7 -- preprocessing & NER)
    # ------------------------------------------------------------------ #
    # The NER model is loaded lazily and cached by the service layer, so a
    # missing model does not prevent the application from starting. The model
    # package itself is installed out-of-band (see README) and is never
    # downloaded automatically by the application or the test suite.
    nlp_model_name: str = "dslim/bert-base-NER"
    nlp_ner_max_length: int = 512
    nlp_max_text_length: int = 10000
    nlp_device: str = "cpu"
    nlp_use_fast_tokenizer: bool = True

    # ------------------------------------------------------------------ #
    # Risk state update (Module 9)
    # ------------------------------------------------------------------ #
    # Risk scores use a 0-100 scale: 0 is the lowest/most benign risk and 100
    # is the highest possible risk. Out-of-range values are rejected as invalid
    # (never silently clamped).
    #
    # Risk levels are derived from the 0-100 score using the thresholds below.
    # Those thresholds are a *configurable assumption* (documented in the
    # README); they are NOT part of an official project specification yet, so
    # they can be tuned via environment variables without code changes:
    #
    #   LOW      [0, 30]                 score < RISK_LEVEL_MEDIUM
    #   MEDIUM   [31, 70]                score < RISK_LEVEL_HIGH
    #   HIGH     [71, 90]                score < RISK_LEVEL_CRITICAL
    #   CRITICAL [91, 100]               score >= RISK_LEVEL_CRITICAL
    risk_score_min: float = 0.0
    risk_score_max: float = 100.0
    risk_level_medium: float = 31.0
    risk_level_high: float = 71.0
    risk_level_critical: float = 91.0

    # ------------------------------------------------------------------ #
    # Risk propagation / ripple effect (Module 10)
    # ------------------------------------------------------------------ #
    # Module 10 starts from a resolved entity that already carries a risk
    # score (set by Module 9) and propagates that risk downstream through the
    # supply-chain graph. The propagated risk of an entity at traversal depth
    # ``d`` is ``source_score * attenuation ** d`` (clamped to the 0-100
    # scale). These defaults are a *configurable assumption* until an official
    # specification defines them; they can be tuned via environment variables
    # without code changes.
    risk_propagation_max_depth: int = 5
    risk_propagation_attenuation: float = 0.5
    risk_propagation_max_affected: int = 500


#: Module-level singleton. Import this everywhere instead of instantiating
#: ``Settings`` repeatedly, to avoid creating multiple settings instances.
settings: Settings = Settings()

