"""Tests for the news ingestion module (Module 6).

These tests do not require a running Neo4j server.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.news import get_ingestion_service
from app.main import app
from app.schemas.news import NewsIngestRequest, NewsIngestResponse
from app.services.nlp.ingestion_service import IngestionService
from app.utils.text_normalizer import compute_content_hash, normalize_text

# ---------------------------------------------------------------------------
# Test setup
# ---------------------------------------------------------------------------

# Set fake Neo4j environment variables before importing the app (required by
# pydantic-settings configuration).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ingestion_service() -> IngestionService:
    return IngestionService()


@pytest.fixture()
def api_client(ingestion_service: IngestionService) -> Any:
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Text normalization tests
# ---------------------------------------------------------------------------


def test_normalize_text_strips_whitespace() -> None:
    assert normalize_text("  hello world  ") == "hello world"


def test_normalize_text_collapses_newlines() -> None:
    assert normalize_text("line1\n\nline2\t\tline3") == "line1 line2 line3"


def test_normalize_text_handles_unicode() -> None:
    assert "café" in normalize_text("café")


def test_normalize_text_removes_null_bytes() -> None:
    assert "\x00" not in normalize_text("hello\x00world")


def test_normalize_text_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalize_text(123)


def test_compute_content_hash_returns_hex() -> None:
    h = compute_content_hash("hello world")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_compute_content_hash_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        compute_content_hash(123)


# ---------------------------------------------------------------------------
# Ingestion service tests
# ---------------------------------------------------------------------------


def test_ingest_returns_success(ingestion_service: IngestionService) -> None:
    payload = NewsIngestRequest(
        title="Test article",
        content="This is the content of the test article.",
        source="Test Source",
        url="https://example.com/article",
        published_at=datetime(2026, 8, 17, 10, 0, 0),
    )

    result = ingestion_service.ingest(payload)

    assert isinstance(result, NewsIngestResponse)
    assert result.status == "ingested"
    assert result.title == "Test article"
    assert result.source == "Test Source"
    assert result.content_length == len("This is the content of the test article.")
    assert result.article_id is not None
    assert result.content_hash is not None


def test_ingest_detects_duplicate(ingestion_service: IngestionService) -> None:
    payload = NewsIngestRequest(
        title="Duplicate article",
        content="Same content here.",
        source="Test Source",
    )

    first = ingestion_service.ingest(payload)
    assert first.status == "ingested"

    second = ingestion_service.ingest(payload)
    assert second.status == "duplicate"


def test_ingest_generates_unique_ids(ingestion_service: IngestionService) -> None:
    payload1 = NewsIngestRequest(title="A", content="Content A")
    payload2 = NewsIngestRequest(title="B", content="Content B")

    result1 = ingestion_service.ingest(payload1)
    result2 = ingestion_service.ingest(payload2)

    assert result1.article_id != result2.article_id


def test_ingest_rejects_invalid_payload(ingestion_service: IngestionService) -> None:
    with pytest.raises(ValueError):
        ingestion_service.ingest("not a payload")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


def test_ingest_endpoint_returns_201(api_client: Any) -> None:
    payload = {
        "title": "Port strike disrupts European shipments",
        "content": "Workers at a major European port announced a strike affecting shipping routes.",
        "source": "Example News",
        "url": "https://example.com/article",
        "published_at": "2026-08-17T10:00:00",
    }

    response = api_client.post("/api/v1/news/ingest", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ingested"
    assert data["title"] == "Port strike disrupts European shipments"
    assert data["source"] == "Example News"
    assert "article_id" in data
    assert "content_hash" in data


def test_ingest_endpoint_missing_title_returns_422(api_client: Any) -> None:
    payload = {
        "content": "Some content without a title.",
    }

    response = api_client.post("/api/v1/news/ingest", json=payload)

    assert response.status_code == 422


def test_ingest_endpoint_missing_content_returns_422(api_client: Any) -> None:
    payload = {
        "title": "Title only",
    }

    response = api_client.post("/api/v1/news/ingest", json=payload)

    assert response.status_code == 422


def test_ingest_endpoint_empty_content_returns_422(api_client: Any) -> None:
    payload = {
        "title": "Valid title",
        "content": "   ",
    }

    response = api_client.post("/api/v1/news/ingest", json=payload)

    assert response.status_code == 422


def test_ingest_endpoint_invalid_url_returns_422(api_client: Any) -> None:
    payload = {
        "title": "Valid title",
        "content": "Valid content.",
        "url": "not-a-valid-url",
    }

    response = api_client.post("/api/v1/news/ingest", json=payload)

    assert response.status_code == 422


def test_ingest_endpoint_excessive_content_returns_422(api_client: Any) -> None:
    payload = {
        "title": "Valid title",
        "content": "x" * 60000,
    }

    response = api_client.post("/api/v1/news/ingest", json=payload)

    assert response.status_code == 422


def test_ingest_endpoint_invalid_timestamp_returns_422(api_client: Any) -> None:
    payload = {
        "title": "Valid title",
        "content": "Valid content.",
        "published_at": "not-a-date",
    }

    response = api_client.post("/api/v1/news/ingest", json=payload)

    assert response.status_code == 422


def test_ingest_endpoint_detects_duplicate(api_client: Any) -> None:
    payload = {
        "title": "Duplicate test",
        "content": "Same duplicate content.",
        "source": "Test",
    }

    response1 = api_client.post("/api/v1/news/ingest", json=payload)
    assert response1.status_code == 201
    assert response1.json()["status"] == "ingested"

    response2 = api_client.post("/api/v1/news/ingest", json=payload)
    assert response2.status_code == 201
    assert response2.json()["status"] == "duplicate"


def test_ingest_endpoint_response_is_json_serializable(api_client: Any) -> None:
    payload = {
        "title": "JSON test",
        "content": "Content for JSON serialization test.",
    }

    response = api_client.post("/api/v1/news/ingest", json=payload)

    assert response.status_code == 201
    response.json()  # must parse without error


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


def test_news_route_is_registered() -> None:
    paths = {r.path for r in app.routes}
    assert "/api/v1/news/ingest" in paths
