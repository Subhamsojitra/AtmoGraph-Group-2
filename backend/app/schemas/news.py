"""Pydantic schemas for the news ingestion API (Module 6).

These models define the request and response contracts for article/text
ingestion. They are intentionally independent of any graph schema or NLP
processing details that will be added in later modules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NewsIngestRequest(BaseModel):
    """Request body for ingesting a news article or text document.

    Only ``title`` and ``content`` are required. The remaining fields are
    optional metadata that improve duplicate detection and downstream
    processing.
    """

    model_config = ConfigDict(extra="ignore")

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Article headline or title",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Full article body text",
    )
    source: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Publisher or data source name",
    )
    url: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Canonical URL of the original article",
    )
    published_at: Optional[datetime] = Field(
        default=None,
        description="Original publication timestamp (ISO 8601)",
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        """Reject whitespace-only titles."""
        if not value.strip():
            raise ValueError("title must not be empty or whitespace-only")
        return value.strip()

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Reject whitespace-only content."""
        if not value.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return value.strip()

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: Optional[str]) -> Optional[str]:
        """Basic URL sanity check when provided."""
        if value is None or value.strip() == "":
            return None
        stripped = value.strip()
        if not stripped.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return stripped


class NewsIngestResponse(BaseModel):
    """Response returned after a successful ingestion request."""

    model_config = ConfigDict(extra="ignore")

    article_id: str = Field(description="Internal unique identifier for the article")
    title: str = Field(description="Article title as ingested")
    source: Optional[str] = Field(default=None, description="Publisher or data source")
    content_length: int = Field(description="Character length of the normalized content")
    status: str = Field(description="Ingestion status, e.g. 'ingested' or 'duplicate'")
    content_hash: Optional[str] = Field(
        default=None, description="SHA-256 hash of the normalized content"
    )
