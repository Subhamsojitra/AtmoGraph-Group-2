"""Risk propagation / ripple effect API routes for AtmoGraph (Module 10).

Exposes an endpoint to propagate a resolved entity's risk downstream through
the supply-chain graph. All business logic lives in
:class:`app.services.risk_propagation.risk_propagation_service.RiskPropagationService`,
which reuses the existing :class:`GraphRepository` and the existing
:class:`RiskService` risk-level engine.

Architecture:

    FastAPI
        ↓
    Risk Propagation API (this module)
        ↓
    RiskPropagationService
        ↓
    GraphRepository (existing) | RiskService (existing)
        ↓
    Neo4jDatabase (existing singleton)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.logger import get_logger
from app.schemas.risk_propagation import (
    RiskPropagationRequest,
    RiskPropagationResponse,
)
from app.services.risk.exceptions import EntityNotFoundError
from app.services.risk_propagation.risk_propagation_service import RiskPropagationService

logger = get_logger(__name__)

router = APIRouter(prefix="/risk-propagation", tags=["risk-propagation"])


# --------------------------------------------------------------------------- #
# Dependency injection
# --------------------------------------------------------------------------- #


def get_risk_propagation_service() -> RiskPropagationService:
    """Provide a fresh :class:`RiskPropagationService` instance to route handlers.

    The service is lightweight and stateless apart from its injected
    :class:`GraphRepository` and :class:`RiskService`, so a new instance per
    request is fine. Tests override this dependency to inject a mocked service.
    """
    return RiskPropagationService()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.post(
    "/propagate",
    response_model=RiskPropagationResponse,
    status_code=status.HTTP_200_OK,
    summary="Propagate a resolved entity's risk downstream",
    description=(
        "Accept a resolved entity (node_id/name and its Module 9 risk score), "
        "traverse the supply-chain graph downstream up to a configurable depth, "
        "and return the affected entities with their propagated risk/impact."
    ),
    responses={
        422: {"description": "Invalid input (invalid risk score, depth, or malformed payload)"},
        404: {"description": "Source entity node not found in graph"},
        503: {"description": "Graph database service unavailable"},
        500: {"description": "Unexpected internal error"},
    },
)
def propagate_risk(
    payload: RiskPropagationRequest,
    service: RiskPropagationService = Depends(get_risk_propagation_service),
) -> RiskPropagationResponse:
    """Propagate a resolved entity's risk downstream through the supply chain.

    Error mapping:
    - 200 + ``propagated=False``: entity is unresolved (no node id)
    - 422: invalid risk score, depth, or attenuation
    - 404: source entity id provided but node does not exist in the graph
    - 503: Neo4j unavailable
    - 500: unexpected internal failure
    """
    try:
        return service.propagate(payload)
    except EntityNotFoundError:
        logger.warning(
            "Risk propagation API: source entity node not found",
            extra={"entity_id": payload.entity_id},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source entity node not found in graph.",
        )
    except ServiceUnavailable:
        logger.error("Risk propagation API: Neo4j service unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Graph database service is currently unavailable. "
                "Please try again later."
            ),
        )
    except Neo4jError as exc:
        logger.error("Risk propagation API: Neo4j error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected database error occurred.",
        )
    except ValueError as exc:
        logger.warning("Risk propagation API: invalid input", extra={"error": str(exc)})
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception:
        logger.error("Risk propagation API: unexpected error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred.",
        )