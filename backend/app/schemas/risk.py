"""Pydantic schemas for the risk state update module (Module 9).

These models define the request and response contracts for updating an existing
resolved entity's risk state in Neo4j. Module 9 never performs entity
recognition or entity resolution again: it consumes the identifier/name of an
entity already resolved by Module 8 and updates only the risk properties of the
existing graph node.

The risk score uses a **0-100 scale** (0 = lowest risk, 100 = highest risk).
Out-of-range values are rejected by validation and are never silently clamped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResolvedEntityInput(BaseModel):
    """Input model representing a resolved entity from Module 8.

    The fields match the expected output of the entity resolution pipeline. The
    risk update service consults ``matched``/``node_id`` to decide whether a
    graph update is permitted at all.
    """

    model_config = ConfigDict(extra="ignore")

    original_text: str = Field(description="Original text as found in the article")
    normalized_text: str = Field(description="Normalized/lowercased entity text")
    ner_label: str = Field(description="NER label, e.g. ORG, GPE, PRODUCT")
    matched: bool = Field(description="Whether the entity was matched to a graph node")
    node_id: Optional[str] = Field(default=None, description="Matched Neo4j node ID")
    node_name: Optional[str] = Field(default=None, description="Matched node display name")
    match_method: Optional[str] = Field(default=None, description="Resolution method used")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Match confidence")


class RiskLevelUpdateRequest(BaseModel):
    """Request to update an existing resolved entity's risk state.

    ``entity_id`` is the Module 8 ``node_id`` for a *matched* entity. A missing,
    null, or blank value means the entity is unresolved, in which case the
    service returns a controlled ``updated=False`` result and never writes to
    Neo4j (no node is ever created).

    ``risk_score`` is required and must be within ``[0, 100]`` inclusive. The
    new ``risk_level`` is derived from the score using the configured
    thresholds.
    """

    model_config = ConfigDict(extra="ignore")

    entity_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description=(
            "Resolved entity identifier (Module 8 node_id). Blank/null means "
            "the entity is unresolved and no graph update is possible."
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
        description="New risk score on the 0-100 scale (inclusive)",
    )
    reason: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Brief, human-readable reason for the risk change",
    )

    @field_validator("entity_id")
    @classmethod
    def normalize_entity_id(cls, value: Optional[str]) -> Optional[str]:
        """Normalize ``entity_id``: whitespace-only values become ``None``.

        A ``None``/blank ``entity_id`` is treated as an unresolved entity by the
        risk service.
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

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: Optional[str]) -> Optional[str]:
        """Normalize ``reason``: whitespace-only values become ``None``."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class RiskStateUpdateResponse(BaseModel):
    """Response returned after a risk state update attempt.

    ``updated`` is ``False`` when the entity was unresolved, the node was not
    found, or Neo4j rejected the write. The ``error`` field is only present on
    failure and never contains credentials/stack traces or secrets.
    """

    model_config = ConfigDict(extra="ignore")

    entity_id: str
    entity_name: Optional[str] = None
    updated: bool
    previous_risk_score: Optional[float] = None
    new_risk_score: Optional[float] = None
    previous_risk_level: Optional[str] = None
    new_risk_level: Optional[str] = None
    reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
