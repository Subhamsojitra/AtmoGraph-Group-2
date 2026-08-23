"""Pydantic schemas for NLP analysis & Named Entity Recognition (Module 7).

These models define the request/response contract for the NER endpoint. They
are intentionally independent of Module 8 entity resolution and the graph
schema: they only describe *clean extracted entities* (text, label, offsets).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.news import NewsIngestRequest


class NEREntity(BaseModel):
    """A single named entity occurrence extracted by the NER model.

    ``start`` and ``end`` are zero-based character offsets into the text that
    was analyzed (``NERAnalysisResponse.analyzed_text``), so that
    ``analyzed_text[start:end]`` equals ``text`` whenever the underlying
    model/tokenizer produces aligned spans.

    ``confidence`` is included because the selected Hugging Face Transformers
    token-classification model genuinely provides a per-entity ``score``. It
    is optional in the schema so alternative backends or tests may omit it.
    """

    model_config = ConfigDict(extra="ignore")

    text: str = Field(description="The extracted entity surface text")
    label: str = Field(description="NER label assigned by the model (e.g. LOC, ORG, PER)")
    start: int = Field(ge=0, description="Zero-based character start offset")
    end: int = Field(ge=0, description="Zero-based character end offset (exclusive)")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Model confidence score, if available"
    )

    @field_validator("end")
    @classmethod
    def validate_end(cls, value: int, info: object) -> int:
        """Guard against inverted/negative span coordinates at the schema level."""
        start = info.data.get("start")
        if start is not None and value < start:
            raise ValueError("end must be greater than or equal to start")
        return value


class NERAnalysisRequest(NewsIngestRequest):
    """Request body for NER analysis of a news article.

    Reuses the Module 6 ``NewsIngestRequest`` fields and their validators
    (title/content validation, URL sanity check, etc.) so no validation logic
    is duplicated.

    ``article_id`` is optional: when the caller has already ingested the
    article via ``POST /api/v1/news/ingest``, the real Module 6 identifier may
    be supplied here so the response references the actual article id. When it
    is omitted, the service generates a fresh identifier using the exact same
    UUIDv4 mechanism as Module 6 ingestion.
    """

    article_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description=(
            "Optional article identifier from a prior Module 6 ingestion. "
            "When omitted, a fresh UUIDv4 identifier is generated."
        ),
    )

    @field_validator("article_id")
    @classmethod
    def validate_article_id(cls, value: Optional[str]) -> Optional[str]:
        """Reject empty/whitespace-only article identifiers."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("article_id must not be empty or whitespace-only")
        return stripped


class NERAnalysisResponse(BaseModel):
    """Structured result of running NER over an article.

    ``entities`` is the occurrence-level list: repeated mentions of the same
    entity are preserved as separate entries with distinct character offsets.
    ``analyzed_text`` is included so that the returned offsets are directly
    interpretable (``analyzed_text[entity.start:entity.end] == entity.text``),
    which is what Module 8 entity resolution will consume.
    """

    model_config = ConfigDict(extra="ignore")

    article_id: str = Field(description="Internal unique identifier for the article")
    analyzed_text: str = Field(
        description=(
            "Normalized text that was analyzed (title + content). Character "
            "offsets in entities refer to this string."
        )
    )
    entities: list[NEREntity] = Field(default_factory=list, description="Extracted entities")
    entity_count: int = Field(ge=0, description="Number of extracted entity occurrences")
