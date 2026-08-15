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
    """Return the current backend service status including Neo4j connectivity."""
    # Check Neo4j connectivity
    neo4j_status = "connected" if neo4j_db.verify_connectivity() else "disconnected"

    # Service status is always healthy if the API is running
    # Neo4j connectivity is reported separately
    return HealthResponse(
        status="healthy",
        service="AtmoGraph API",
        neo4j=neo4j_status
    )