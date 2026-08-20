"""AtmoGraph FastAPI application entrypoint.

Run locally with:

    uvicorn app.main:app --reload

Interactive API documentation is served at ``/docs`` and the OpenAPI schema is
available at ``/openapi.json``.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.entity_resolution import router as entity_resolution_router
from app.api.graph import router as graph_router
from app.api.health import router as health_router
from app.api.news import router as news_router
from app.database.neo4j import neo4j_db

logger = logging.getLogger("app.main")

APP_NAME = "AtmoGraph API"
APP_VERSION = "0.1.0"
API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events handler.

    Startup: initialize the shared (reusable) Neo4j driver and probe the real
    database connectivity. A missing/unreachable Neo4j is logged but does not
    prevent the application from starting; the health endpoint reports the
    accurate status.

    Shutdown: close the driver and release all resources.
    """
    # Startup
    neo4j_db.initialize()
    if neo4j_db.verify_connectivity():
        logger.info("Neo4j connectivity at startup: connected")
    else:
        logger.warning(
            "Neo4j connectivity at startup: disconnected - the /health endpoint "
            "will report 'disconnected' until Neo4j is reachable"
        )
    yield
    # Shutdown
    await neo4j_db.close_async()


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Supply Chain Ripple Effect Predictor - Backend API.",
    lifespan=lifespan,
)

app.include_router(entity_resolution_router, prefix=API_PREFIX)
app.include_router(health_router, prefix=API_PREFIX)
app.include_router(graph_router, prefix=API_PREFIX)
app.include_router(news_router, prefix=API_PREFIX)
