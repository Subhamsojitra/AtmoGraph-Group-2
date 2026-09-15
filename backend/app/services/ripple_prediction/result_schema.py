"""Result schemas for the ripple prediction service (Module 17, Part 3).

Defines the structured outcome of a ripple prediction operation. These models
are transport-agnostic Pydantic schemas that the WebSocket layer will later
serialize into ``ripple_prediction_result`` envelopes.

Only fields actually produced by the system are included: affected entities
carry the real propagated risk from the existing Module 10 engine and the
real GNN prediction (when available) from the existing Module 14 engine.
Nothing is fabricated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RippleAffectedEntity(BaseModel):
    """A single entity impacted downstream by the source entity's risk.

    ``propagated_risk_score`` / ``propagated_risk_level`` come from the
    existing Module 10 risk propagation engine. ``gnn_prediction`` is the
    real Module 14 GNN prediction for this entity when the entity is part
    of the current graph served by the GNN; ``None`` otherwise. No
    prediction value is ever invented.
    """

    model_config = ConfigDict(extra="ignore")

    entity_id: str = Field(
        description="Neo4j node identifier of the affected entity"
    )
    entity_name: Optional[str] = Field(
        default=None, description="Display name of the affected entity"
    )
    depth: int = Field(
        ge=1, description="Traversal depth (hops) from the source entity"
    )
    propagated_risk_score: float = Field(
        ge=0.0, le=100.0,
        description="Propagated risk score on the 0-100 scale (Module 10)",
    )
    propagated_risk_level: str = Field(
        description=(
            "Propagated risk level derived with the existing risk thresholds"
        )
    )
    gnn_prediction: Optional[float] = Field(
        default=None,
        description=(
            "Real GNN prediction for this entity from the existing Module 14 "
            "pipeline. None when the entity has no prediction in the current "
            "graph or prediction is unavailable. Never fabricated."
        ),
    )


class RipplePredictionResult(BaseModel):
    """Structured result of a ripple prediction operation (Module 17).

    Produced by :class:`RipplePredictionService` from the outputs of the
    existing risk propagation and GNN prediction services. JSON-serializable
    and free of any transport concerns.
    """

    model_config = ConfigDict(extra="ignore")

    source_entity_id: str = Field(
        description="Neo4j node identifier of the source entity"
    )
    source_entity_name: Optional[str] = Field(
        default=None, description="Display name of the source entity"
    )
    source_risk_score: float = Field(
        ge=0.0, le=100.0,
        description="Risk score of the source entity on the 0-100 scale",
    )
    propagated: bool = Field(
        description=(
            "True when risk propagation ran (the source entity exists). "
            "False only when the entity was unresolved."
        )
    )
    affected_entities: list[RippleAffectedEntity] = Field(
        default_factory=list,
        description="Entities affected downstream by the source risk",
    )
    affected_count: int = Field(
        ge=0, description="Number of affected entities"
    )
    max_depth_reached: int = Field(
        ge=0, description="Maximum propagation depth that was reached"
    )
    prediction_count: int = Field(
        ge=0,
        description=(
            "Number of GNN predictions obtained for affected entities "
            "(0 when prediction is unavailable)"
        ),
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time at which the ripple prediction was computed",
    )
    error: Optional[str] = Field(
        default=None,
        description=(
            "Controlled, client-safe error description when propagation "
            "could not run. Never contains stack traces or secrets."
        ),
    )
