"""Pydantic schemas for Entity Resolution & Neo4j Mapping (Module 8).

These models define the request/response contracts for entity resolution,
mapping extracted NER entities to candidate graph nodes in Neo4j without
assuming fixed node schema properties or mutating the graph.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.ner import NEREntity
from app.schemas.news import NewsIngestRequest


class EntityCandidate(BaseModel):
    """A candidate Neo4j graph node matching a given entity."""

    model_config = ConfigDict(extra="ignore")

    node_id: str = Field(description="Neo4j node identifier")
    node_name: str = Field(description="Display name or primary label of the graph node")
    node_label: str = Field(description="Neo4j node label (e.g., Port, Supplier, Location)")
    score: float = Field(ge=0.0, le=1.0, description="Matching similarity score")
    match_type: str = Field(description="Candidate match classification: exact, alias, fuzzy")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Raw node properties from Neo4j"
    )


class ResolvedEntity(BaseModel):
    """Result of attempting to resolve a single NER entity occurrence to a Neo4j node."""

    model_config = ConfigDict(extra="ignore")

    original_text: str = Field(description="Original surface text extracted by NER")
    normalized_text: str = Field(description="Normalized comparison form of the entity text")
    ner_label: str = Field(description="NER label from Module 7 (e.g., ORG, LOC, PER)")
    matched: bool = Field(description="True if entity was successfully resolved to a Neo4j node")
    node_id: Optional[str] = Field(
        default=None, description="Matched Neo4j node ID if matched is True"
    )
    node_label: Optional[str] = Field(
        default=None, description="Matched Neo4j node label if matched is True"
    )
    node_name: Optional[str] = Field(
        default=None, description="Matched Neo4j node primary name if matched is True"
    )
    match_method: str = Field(
        description="Resolution status: exact, alias, fuzzy, unresolved, ambiguous"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence score for the resolution decision"
    )
    candidates: Optional[list[EntityCandidate]] = Field(
        default=None, description="Top matching node candidates evaluated during resolution"
    )


class NERResolutionRequest(BaseModel):
    """Request payload containing pre-extracted NER entities to resolve."""

    model_config = ConfigDict(extra="ignore")

    article_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional article identifier associated with these entities",
    )
    entities: list[NEREntity] = Field(
        description="List of extracted NER entities to resolve against Neo4j"
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


class ArticleResolutionRequest(NewsIngestRequest):
    """Request payload containing raw article text to analyze (NER) and resolve in one flow."""

    article_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional article identifier from a prior ingestion",
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


class EntityResolutionResponse(BaseModel):
    """Structured result of resolving NER entities against Neo4j graph nodes."""

    model_config = ConfigDict(extra="ignore")

    article_id: str = Field(description="Internal unique identifier for the article/request")
    resolved_entities: list[ResolvedEntity] = Field(
        default_factory=list, description="Per-entity resolution results"
    )
    total_entities: int = Field(ge=0, description="Total number of entity occurrences submitted")
    resolved_count: int = Field(ge=0, description="Number of successfully matched entities")
    unresolved_count: int = Field(ge=0, description="Number of unresolved entities")
    ambiguous_count: int = Field(ge=0, description="Number of ambiguous entities")
