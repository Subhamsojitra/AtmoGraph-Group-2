"""Health check router.

Exposes a lightweight endpoint used by orchestrators, load balancers and
monitoring tools to confirm the backend service is up and responding.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Schema returned by the health check endpoint."""

    status: str
    service: str


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return the current backend service status.

    Module 1 only reports liveness of the running FastAPI process. Database /
    graph connectivity checks are introduced in a later module.
    """
    return HealthResponse(status="healthy", service="AtmoGraph API")