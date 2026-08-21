"""Health check router.

Exposes a lightweight endpoint used by orchestrators, load balancers and
monitoring tools to confirm the backend service is up and responding.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.database.neo4j import neo4j_db

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Schema returned by the health check endpoint."""

    status: str
    service: str
    neo4j: str


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return the current backend service status including Neo4j connectivity.

    The Neo4j status is the result of a *real* connectivity check
    (``RETURN 1``) -- it is never assumed or faked. When Neo4j is
    unreachable the service is reported as ``degraded``, not ``healthy``.
    """
    # Check Neo4j connectivity (performs an actual round-trip query)
    neo4j_connected = neo4j_db.verify_connectivity()

    # The API is up; Neo4j dependency status is accurate and separate.
    return HealthResponse(
        status="healthy" if neo4j_connected else "degraded",
        service="AtmoGraph API",
        neo4j="connected" if neo4j_connected else "disconnected",
    )