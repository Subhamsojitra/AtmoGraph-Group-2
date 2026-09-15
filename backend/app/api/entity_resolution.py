"""Entity Resolution API routes for AtmoGraph (Module 8).

Exposes an endpoint for resolving NER entities against the existing Neo4j
graph. All business logic lives in
:class:`app.services.entity_resolution.entity_resolution_service.EntityResolutionService`.

Architecture:

    FastAPI
        ↓
    Entity Resolution API (this module)
        ↓
    EntityResolutionService
        ↓
    GraphRepository
        ↓
    Neo4jDatabase
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from neo4j.exceptions import ServiceUnavailable

from app.core.logger import get_logger
from app.schemas.entity_resolution import EntityResolutionResponse, NERResolutionRequest
from app.services.entity_resolution.entity_resolution_service import EntityResolutionService
from app.services.entity_resolution.exceptions import EntityResolutionError

logger = get_logger(__name__)

router = APIRouter(prefix="/news", tags=["news"])


# --------------------------------------------------------------------------- #
# Dependency injection
# --------------------------------------------------------------------------- #


def get_entity_resolution_service() -> EntityResolutionService:
    """Provide a fresh :class:`EntityResolutionService` instance to route handlers."""
    return EntityResolutionService()


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def _call_service(operation: Any) -> Any:
    """Execute a service call and map domain errors to HTTP responses."""
    try:
        return operation()
    except (EntityResolutionError,):
        logger.error("Entity resolution failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during entity resolution.",
        )
    except Exception as exc:
        logger.error("Entity resolution unexpected error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal error occurred.",
        )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.post(
    "/resolve",
    response_model=EntityResolutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Resolve NER entities against the graph",
    description=(
        "Accept a batch of pre-extracted NER entities, search the Neo4j graph "
        "for candidate nodes, and return structured resolution results (exact, "
        "fuzzy, ambiguous, or unresolved). Does not create or mutate graph data."
    ),
    responses={
        422: {"description": "Invalid input (empty entities list or malformed payload)"},
        503: {"description": "Graph database is currently unavailable"},
        500: {"description": "Unexpected resolution failure"},
    },
)
def resolve_entities(
    payload: NERResolutionRequest,
    service: EntityResolutionService = Depends(get_entity_resolution_service),
) -> EntityResolutionResponse:
    """Resolve NER entities to Neo4j graph nodes.

    Error mapping:
    - 422: invalid input (``ValueError``)
    - 503: Neo4j unreachable (``ServiceUnavailable``)
    - 500: unexpected failure
    """
    try:
        return service.resolve(
            entities=payload.entities,
            article_id=payload.article_id,
        )
    except ValueError as exc:
        logger.warning("Entity resolution validation error", extra={"error": str(exc)})
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ServiceUnavailable:
        logger.error("Entity resolution API: Neo4j unavailable")
        raise HTTPException(
            status_code=503,
            detail=(
                "Graph database service is currently unavailable. "
                "Please try again later."
            ),
        )
    except EntityResolutionError:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during entity resolution.",
        )
    except Exception:
        logger.error("Entity resolution API: unexpected error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal error occurred.",
        )
