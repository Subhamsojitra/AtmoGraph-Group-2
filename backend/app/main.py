"""AtmoGraph FastAPI application entrypoint.

Run locally with:

    uvicorn app.main:app --reload

Interactive API documentation is served at ``/docs`` and the OpenAPI schema is
available at ``/openapi.json``.
"""

from fastapi import FastAPI

from app.api.health import router as health_router

APP_NAME = "AtmoGraph API"
APP_VERSION = "0.1.0"
API_PREFIX = "/api/v1"

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Supply Chain Ripple Effect Predictor - Backend API.",
)

app.include_router(health_router, prefix=API_PREFIX)
