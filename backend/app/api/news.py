"""News ingestion & NLP analysis API routes for AtmoGraph (Modules 6 & 7).

This module exposes endpoints for ingesting external news articles and text
documents (Module 6) and for running Named Entity Recognition over article
text (Module 7). It delegates all business logic to the service layer
(:class:`app.services.nlp.ingestion_service.IngestionService` and
:class:`app.services.nlp.ner_service.NERService`) and does not directly
access Neo4j or run NLP models itself.

Architecture:

    FastAPI
        ↓
    News/NLP API (this module)
        ↓
    IngestionService | NERService
        ↓
    Text Normalization | Transformers NER model
        ↓
    Structured Ingestion/NER Result
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.logger import get_logger
from app.schemas.ner import NERAnalysisRequest, NERAnalysisResponse
from app.schemas.news import NewsIngestRequest, NewsIngestResponse
from app.services.nlp.exceptions import (
    ModelUnavailableError,
    NERProcessingError,
    TextPreprocessingError,
)
from app.services.nlp.ingestion_service import IngestionService
from app.services.nlp.ner_service import NERService

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


# The NER service owns the heavy NLP model, which must be loaded exactly once
# and reused across all requests (never per request). A module-level singleton
# is therefore appropriate here, unlike the stateless IngestionService.
_ner_service = NERService()


def get_ner_service() -> NERService:
    """Provide the shared :class:`NERService` singleton to route handlers.

    The model is loaded lazily by :class:`NERModelManager` on first use and
    then cached for the lifetime of the process. Tests override this
    dependency with a lightweight fake model manager.
    """
    return _ner_service


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


@router.post(
    "/analyze",
    response_model=NERAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze named entities in a news article",
    description=(
        "Normalize article text and run Named Entity Recognition. Returns "
        "structured entity occurrences (text, label, character offsets, "
        "confidence) with no database writes. Accepts an optional "
        "article_id from a prior ingestion so the response can reference the "
        "real Module 6 identifier."
    ),
    responses={
        422: {"description": "Invalid input (empty, whitespace-only, or oversized text)"},
        503: {"description": "NLP model is currently unavailable"},
        500: {"description": "Unexpected NLP processing failure"},
    },
)
def analyze_article(
    payload: NERAnalysisRequest,
    service: NERService = Depends(get_ner_service),
) -> NERAnalysisResponse:
    """Run NER over a news article and return structured entity results.

    Error mapping (no stack traces or internals are exposed):
    - 422: invalid/missing/oversized input (``TextPreprocessingError``/``ValueError``)
    - 503: NLP model unavailable (``ModelUnavailableError``)
    - 500: unexpected processing failure (``NERProcessingError`` or unknown)
    """
    try:
        return service.analyze(payload)
    except TextPreprocessingError as exc:
        logger.warning("NER preprocessing error", extra={"error": str(exc)})
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("NER validation error", extra={"error": str(exc)})
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailableError:
        logger.error("NLP model unavailable during analysis")
        raise HTTPException(
            status_code=503,
            detail="The NLP model is currently unavailable. Please try again later.",
        )
    except NERProcessingError:
        logger.error("NER processing failed")
        raise HTTPException(
            status_code=500,
            detail="NLP processing failed unexpectedly.",
        )
    except Exception:
        logger.error("NER analysis failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during NLP analysis.",
        )

