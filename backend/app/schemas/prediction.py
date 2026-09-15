"""Pydantic schemas for the GNN prediction module (Module 14).

These models define the request/response contract for serving node-level
downstream-delay predictions from the trained GNN (Modules 12/13) over the
current supply-chain graph (prepared by Module 11).

The contract is deliberately minimal and stable:

* ``predictions[i].node_id`` is the graph entity identifier — the frontend
  maps it to its graph node id (``prediction.nodeId``).
* ``predictions[i].prediction`` is the RAW model output (a scalar regression
  value). The project specification defines NO severity thresholds for
  predictions, so NO HIGH/MEDIUM/LOW classification is derived here — the
  raw value is served as-is and interpretation is left to later modules.

The prediction target is node-level regression exactly as in Modules 12/13;
there is no classification, no graph-level pooling and no accuracy claim.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    """Optional request body for the prediction endpoint.

    All fields are optional; an empty body means "predict for the current
    graph with default settings". ``relationship_types`` mirrors the Module
    11 dataset builder: when omitted, ALL outgoing relationships are used
    for feature/edge preparation; when provided, only the listed
    relationship types are considered supply-chain flow (Module 10's
    convention, e.g. ``["SUPPLIES"]``).
    """

    model_config = ConfigDict(extra="ignore")

    relationship_types: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional relationship types that represent supply-chain flow "
            "for dataset preparation (Module 11). Omitted = all outgoing "
            "relationships."
        ),
    )

    @field_validator("relationship_types")
    @classmethod
    def clean_relationship_types(
        cls, value: Optional[list[str]]
    ) -> Optional[list[str]]:
        """Drop blank entries; an all-blank list becomes ``None``."""
        if value is None:
            return None
        cleaned = [
            t.strip() for t in value if isinstance(t, str) and t.strip()
        ]
        return cleaned or None


class NodePrediction(BaseModel):
    """The raw model prediction for a single graph entity."""

    model_config = ConfigDict(extra="ignore")

    node_id: str = Field(description="Neo4j node identifier of the entity")
    prediction: float = Field(
        description=(
            "Raw scalar model output (node-level regression, e.g. predicted "
            "downstream delay). No severity classification is derived."
        ),
        allow_inf_nan=False,
    )


class PredictionResponse(BaseModel):
    """One prediction per graph node, in dataset (node id) order.

    ``prediction_count`` always equals ``len(predictions)`` — one entry per
    graph node, no graph-level pooling.
    """

    model_config = ConfigDict(extra="ignore")

    predictions: list[NodePrediction] = Field(default_factory=list)
    prediction_count: int = Field(
        ge=0,
        description="Number of predictions (one per graph node)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time at which the inference was served",
    )