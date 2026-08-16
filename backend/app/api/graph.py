"""Graph API routes for AtmoGraph.

This module exposes generic, schema-neutral graph operations over HTTP.
It depends on :class:`app.services.graph_service.GraphService` for all
business logic and must not contain Cypher queries or create Neo4j
sessions/drivers directly.

Architecture:

    FastAPI
        ↓
    Graph API (this module)
        ↓
    GraphService
        ↓
    GraphRepository
        ↓
    Neo4jDatabase

Error handling: the service layer may raise ``ServiceUnavailable`` (Neo4j
unreachable), ``Neo4jError`` (database error) or ``ValueError`` (invalid
input). These are translated here into HTTP responses without leaking raw
stack traces or credentials. All responses are JSON serializable Pydantic
models.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.logger import get_logger
from app.schemas.graph import (
    GraphNodeCreateRequest,
    GraphNodeResponse,
    GraphRelationshipCreateRequest,
    GraphRelationshipResponse,
)
from app.services.graph_service import GraphService

logger = get_logger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])


# --------------------------------------------------------------------------- #
# Dependency injection
# --------------------------------------------------------------------------- #


def get_graph_service() -> GraphService:
    """Provide a :class:`GraphService` instance to route handlers.

    Routes depend on this callable so that tests can override it via
    ``app.dependency_overrides`` and inject a mocked service without a
    running Neo4j server.
    """
    return GraphService()


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def _call_service(operation: Any) -> Any:
    """Execute a service call and map domain errors to HTTP responses.

    ``ServiceUnavailable`` (Neo4j unreachable) and ``Neo4jError`` are raised
    by the service when the database is down or fails. ``ValueError`` signals
    invalid input. Any other unexpected exception is logged and returned as a
    generic 500 so that raw stack traces or credentials are never leaked.
    """
    try:
        return operation()
    except ServiceUnavailable:
        logger.error("Graph API: Neo4j service unavailable")
        raise HTTPException(
            status_code=503,
            detail=(
                "Graph database service is currently unavailable. "
                "Please try again later."
            ),
        )
    except Neo4jError as exc:
        logger.error("Graph API: Neo4j error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=500,
            detail="An unexpected database error occurred.",
        )
    except ValueError as exc:
        logger.warning("Graph API: invalid input", extra={"error": str(exc)})
        raise HTTPException(status_code=422, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Graph API: unexpected error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal error occurred.",
        )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.get(
    "/nodes/{node_id}",
    response_model=GraphNodeResponse,
    summary="Get a node by ID",
)
def get_node_by_id_endpoint(
    node_id: str,
    service: GraphService = Depends(get_graph_service),
) -> GraphNodeResponse:
    """Retrieve a single graph node by its identifier.

    Returns ``404`` if no node with the given ``node_id`` exists.
    """
    node = _call_service(lambda: service.get_node_by_id(node_id))
    if node is None:
        raise HTTPException(
            status_code=404,
            detail=f"Node with id '{node_id}' was not found",
        )
    return node


@router.get(
    "/nodes",
    response_model=list[GraphNodeResponse],
    summary="List graph nodes",
)
def get_nodes_endpoint(
    limit: int = Query(
        default=100, ge=1, le=1000, description="Maximum number of nodes to return"
    ),
    service: GraphService = Depends(get_graph_service),
) -> list[GraphNodeResponse]:
    """Retrieve a page of generic graph nodes."""
    return _call_service(lambda: service.get_nodes(limit=limit))


@router.get(
    "/search",
    response_model=list[GraphNodeResponse],
    summary="Search graph nodes",
)
def search_nodes_endpoint(
    label: Optional[str] = Query(default=None, description="Optional node label filter"),
    properties: Optional[str] = Query(
        default=None,
        description='Optional JSON-encoded property filter, e.g. {"name": "A"}',
    ),
    limit: int = Query(
        default=100, ge=1, le=1000, description="Maximum number of results"
    ),
    service: GraphService = Depends(get_graph_service),
) -> list[GraphNodeResponse]:
    """Search nodes by optional label and property filters."""
    properties_dict: Optional[dict[str, Any]] = None
    if properties:
        try:
            parsed = json.loads(properties)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422,
                detail="'properties' must be a valid JSON object",
            ) from e
        if not isinstance(parsed, dict):
            raise HTTPException(
                status_code=422,
                detail="'properties' must be a JSON object",
            )
        properties_dict = parsed

    return _call_service(
        lambda: service.find_nodes(
            label=label, properties=properties_dict, limit=limit
        )
    )


@router.get(
    "/nodes/{node_id}/neighbors",
    response_model=list[GraphNodeResponse],
    summary="Get a node's neighbors",
)
def get_neighbors_endpoint(
    node_id: str,
    relationship_types: Optional[str] = Query(
        default=None,
        description="Comma-separated relationship type filters",
    ),
    limit: int = Query(
        default=100, ge=1, le=1000, description="Maximum number of neighbors"
    ),
    service: GraphService = Depends(get_graph_service),
) -> list[GraphNodeResponse]:
    """Retrieve neighboring nodes connected to the given node."""
    rel_types: Optional[list[str]] = None
    if relationship_types:
        rel_types = [t.strip() for t in relationship_types.split(",") if t.strip()]
        if not rel_types:
            raise HTTPException(
                status_code=422,
                detail="'relationship_types' must contain at least one type",
            )
    return _call_service(
        lambda: service.find_neighbors(
            node_id=node_id,
            relationship_types=rel_types,
            limit=limit,
        )
    )


@router.post(
    "/nodes",
    response_model=GraphNodeResponse,
    status_code=201,
    summary="Create a generic graph node",
)
def create_node_endpoint(
    payload: GraphNodeCreateRequest,
    service: GraphService = Depends(get_graph_service),
) -> GraphNodeResponse:
    """Create a generic, schema-neutral graph node."""
    return _call_service(
        lambda: service.create_node(
            label=payload.label, properties=payload.properties
        )
    )


@router.post(
    "/relationships",
    response_model=GraphRelationshipResponse,
    status_code=201,
    summary="Create a generic graph relationship",
)
def create_relationship_endpoint(
    payload: GraphRelationshipCreateRequest,
    service: GraphService = Depends(get_graph_service),
) -> GraphRelationshipResponse:
    """Create a generic, schema-neutral relationship between two nodes."""
    return _call_service(
        lambda: service.create_relationship(
            source_id=payload.source_id,
            target_id=payload.target_id,
            rel_type=payload.rel_type,
            properties=payload.properties,
        )
    )
