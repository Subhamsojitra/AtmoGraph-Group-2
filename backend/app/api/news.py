"""News ingestion API routes for AtmoGraph (Module 6).

This module exposes endpoints for ingesting external news articles and text
documents. It delegates all business logic to
:class:`app.services.nlp.ingestion_service.IngestionService` and does not
directly access Neo4j or perform NLP processing.

Architecture:

    FastAPI
        ↓
    News API (this module)
        ↓
    IngestionService
        ↓
    Text Normalization
        ↓
    Structured Ingestion Result
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.logger import get_logger
from app.schemas.news import NewsIngestRequest, NewsIngestResponse
from app.services.nlp.ingestion_service import IngestionService

logger = get_logger(__name__)

router = APIRouter(prefix="/news", tags=["news"])


# --------------------------------------------------------------------------- #
# Dependency injection
# --------------------------------------------------------------------------- #


def get_ingestion_service() -> IngestionService:
    """Provide a fresh :class:`IngestionService` instance to route handlers.

    The service is intentionally lightweight and stateless (apart from
    in-memory duplicate tracking), so a new instance per request is
    acceptable for Module 6. Tests can override this dependency.
    """
    return IngestionService()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.post(
    "/ingest",
    response_model=NewsIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a news article",
    description=(
        "Accept a news article or text document, normalize it, assign an "
        "internal identifier, and return a structured ingestion result."
    ),
)
def ingest_article(
    payload: NewsIngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> NewsIngestResponse:
    """Ingest a news article and return its processed representation."""
    try:
        return service.ingest(payload)
    except ValueError as exc:
        logger.warning("News ingestion validation error", extra={"error": str(exc)})
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("News ingestion failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during ingestion.",
        ) from exc
