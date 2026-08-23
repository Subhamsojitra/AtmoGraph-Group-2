"""Pydantic schemas for the risk propagation / ripple effect module (Module 10).

These models define the request/response contracts for propagating a resolved
entity's risk state (produced by Module 9) downstream through the supply-chain
graph. Module 10 never performs entity recognition, entity resolution, or risk
state updates: it consumes an entity identifier/name and a risk score already
established upstream and only *reads* the graph to compute the ripple effect.

The propagated risk uses the same 0-100 scale as Module 9. Risk levels are
derived with the *existing* risk-level thresholds (see :mod:`app.schemas.risk`
and :class:`app.services.risk.risk_service.RiskService`); Module 10 does not
duplicate risk-calculation logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskPropagationRequest(BaseModel):
    """Request to propagate a resolved entity's risk downstream in the graph.

    ``entity_id`` is the Module 8 ``node_id`` for a *matched* entity. A missing,
    null, or blank value means the entity is unresolved, in which case the
    service returns a controlled ``propagated=False`` result and never reads the
    database.

    ``risk_score`` is required and must be within ``[0, 100]`` inclusive (this
    is the score already set for the entity by Module 9).

    ``max_depth`` and ``attenuation`` are optional; when omitted the service
    falls back to the configurable defaults (``RISK_PROPAGATION_*``).
    """

    model_config = ConfigDict(extra="ignore")

    entity_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description=(
            "Resolved entity identifier (Module 8 node_id). Blank/null means "
            "the entity is unresolved and no propagation is possible."
        ),
    )
    entity_name: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Resolved entity display name (Module 8 node_name)",
    )
    risk_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Risk score of the source entity on the 0-100 scale (Module 9)",
    )
    max_depth: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum traversal depth (hops). Defaults to the configured value.",
    )
    attenuation: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Per-hop attenuation factor applied to the source risk score. "
            "Defaults to the configured value."
        ),
    )
    relationship_types: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional relationship type filters to follow downstream. When "
            "omitted, all outgoing relationship types are traversed."
        ),
    )

    @field_validator("entity_id")
    @classmethod
    def normalize_entity_id(cls, value: Optional[str]) -> Optional[str]:
        """Normalize ``entity_id``: whitespace-only values become ``None``.

        A ``None``/blank ``entity_id`` is treated as an unresolved entity by the
        risk propagation service.
        """
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("entity_name")
    @classmethod
    def normalize_entity_name(cls, value: Optional[str]) -> Optional[str]:
        """Normalize ``entity_name``: whitespace-only values become ``None``."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("relationship_types")
    @classmethod
    def clean_relationship_types(
        cls, value: Optional[list[str]]
    ) -> Optional[list[str]]:
        """Drop blank entries and blank/whitespace-only relationship types."""
        if value is None:
            return None
        cleaned = [
            t.strip()
            for t in value
            if isinstance(t, str) and t.strip()
        ]
        return cleaned or None


class AffectedEntity(BaseModel):
    """A single entity impacted downstream by the source entity's risk."""

    model_config = ConfigDict(extra="ignore")

    entity_id: str = Field(description="Neo4j node identifier of the affected entity")
    entity_name: Optional[str] = Field(
        default=None, description="Display name of the affected entity"
    )
    depth: int = Field(ge=1, description="Traversal depth (hops) from the source entity")
    propagated_risk_score: float = Field(
        ge=0.0, le=100.0, description="Propagated risk score on the 0-100 scale"
    )
    propagated_risk_level: str = Field(
        description="Propagated risk level derived with the existing risk thresholds"
    )


class RiskPropagationResponse(BaseModel):
    """Result of propagating a resolved entity's risk downstream in the graph.

    ``propagated`` is ``False`` only when the entity was unresolved, in which
    case no database access is performed and an explanatory ``error`` is set.
    When the source entity exists, ``propagated`` is ``True`` even if no
    downstream entities were found (in which case ``affected_entities`` is
    empty).
    """

    model_config = ConfigDict(extra="ignore")

    source_entity_id: str
    source_entity_name: Optional[str] = None
    source_risk_score: float
    propagated: bool
    affected_entities: list[AffectedEntity] = Field(default_factory=list)
    affected_count: int = Field(ge=0)
    max_depth_reached: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
