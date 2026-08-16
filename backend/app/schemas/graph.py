"""Generic graph response schemas for AtmoGraph.

These schemas are intentionally schema-neutral so that the finalized
supply-chain node and relationship model can be integrated later without
breaking the API contract.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class GraphNodeResponse(BaseModel):
    """Generic graph node response.

    The ``properties`` mapping holds node attributes without assuming
    any specific field names. The finalized schema will populate this
    with concrete fields (e.g., supplier, factory, warehouse).
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    label: str
    properties: dict[str, Any]


class GraphRelationshipResponse(BaseModel):
    """Generic graph relationship response."""

    model_config = ConfigDict(extra="ignore")

    id: str
    type: str
    source_id: str
    target_id: str
    properties: dict[str, Any] = {}


class GraphResultResponse(BaseModel):
    """Wrapper for a single graph query result."""

    model_config = ConfigDict(extra="ignore")

    data: dict[str, Any]


class GraphQueryResponse(BaseModel):
    """Wrapper for a paginated or bulk graph query result."""

    model_config = ConfigDict(extra="ignore")

    results: list[GraphResultResponse]
    count: int


class GraphNodeCreateRequest(BaseModel):
    """Request body for creating a generic graph node."""

    model_config = ConfigDict(extra="ignore")

    label: str
    properties: dict[str, Any] = {}


class GraphRelationshipCreateRequest(BaseModel):
    """Request body for creating a generic graph relationship."""

    model_config = ConfigDict(extra="ignore")

    source_id: str
    target_id: str
    rel_type: str
    properties: dict[str, Any] = {}


class GraphSearchRequest(BaseModel):
    """Request body for searching graph nodes."""

    model_config = ConfigDict(extra="ignore")

    label: Optional[str] = None
    properties: Optional[dict[str, Any]] = None
    limit: int = 100
