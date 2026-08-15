"""AtmoGraph FastAPI application entrypoint.

Run locally with:

    uvicorn app.main:app --reload

Interactive API documentation is served at ``/docs`` and the OpenAPI schema is
available at ``/openapi.json``.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.database.neo4j import neo4j_db

APP_NAME = "AtmoGraph API"
APP_VERSION = "0.1.0"
API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events handler.

    Handles startup and shutdown events for the application.
    """
    # Startup
    neo4j_db.initialize()
    yield
    # Shutdown
    await neo4j_db.close_async()


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Supply Chain Ripple Effect Predictor - Backend API.",
    lifespan=lifespan,
)

app.include_router(health_router, prefix=API_PREFIX)
