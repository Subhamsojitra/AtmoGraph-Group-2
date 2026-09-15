"""NLP text ingestion service for AtmoGraph (Module 6).

This module implements the ingestion pipeline for external news articles and
text documents. It is responsible for:

1. Validating incoming request payloads.
2. Normalizing title and content text.
3. Generating internal article identifiers.
4. Detecting duplicates.
5. Returning structured ingestion results.

Important: this service does NOT perform NER, entity resolution, risk
calculation, or Neo4j updates. Those belong to later modules.
"""

from __future__ import annotations

import uuid
from typing import Optional

from app.core.logger import get_logger
from app.schemas.news import NewsIngestRequest, NewsIngestResponse
from app.utils.text_normalizer import compute_content_hash, normalize_text

logger = get_logger(__name__)


class IngestionService:
    """Service that ingests raw text articles into a normalized internal form.

    Duplicate detection is performed in-memory for Module 6. A production
    implementation should replace the internal ``_seen_hashes`` set with a
    persistent store (e.g., Redis or the graph database) once the schema
    is finalized.
    """

    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()

    def ingest(self, payload: NewsIngestRequest) -> NewsIngestResponse:
        """Process an ingestion request and return a structured result.

        Args:
            payload: Validated request model containing article metadata.

        Returns:
            A ``NewsIngestResponse`` describing the ingestion outcome.

        Raises:
            ValueError: If the payload is invalid or processing fails.
        """
        if not isinstance(payload, NewsIngestRequest):
            raise ValueError("payload must be a NewsIngestRequest instance")

        logger.info(
            "Ingestion request received",
            extra={
                "title": payload.title,
                "source": payload.source,
                "url": payload.url,
            },
        )

        normalized_title = normalize_text(payload.title)
        normalized_content = normalize_text(payload.content)
        content_hash = compute_content_hash(normalized_content)

        logger.debug(
            "Text normalized",
            extra={
                "content_length": len(normalized_content),
                "content_hash": content_hash,
            },
        )

        if content_hash in self._seen_hashes:
            logger.info(
                "Duplicate article detected",
                extra={"content_hash": content_hash, "title": normalized_title},
            )
            return NewsIngestResponse(
                article_id=self._generate_article_id(),
                title=normalized_title,
                source=payload.source,
                content_length=len(normalized_content),
                status="duplicate",
                content_hash=content_hash,
            )

        self._seen_hashes.add(content_hash)

        article_id = self._generate_article_id()

        logger.info(
            "Article ingested successfully",
            extra={"article_id": article_id, "title": normalized_title},
        )

        return NewsIngestResponse(
            article_id=article_id,
            title=normalized_title,
            source=payload.source,
            content_length=len(normalized_content),
            status="ingested",
            content_hash=content_hash,
        )

    def _generate_article_id(self) -> str:
        """Generate a unique internal article identifier.

        Uses UUIDv4 to avoid exposing content details in the identifier
        while ensuring global uniqueness.

        Returns:
            UUIDv4 string identifier.
        """
        return str(uuid.uuid4())
