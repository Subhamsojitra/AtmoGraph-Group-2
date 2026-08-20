"""Unit tests for Entity Resolution (Module 8).

These tests mock the GraphRepository so they do not require a running
Neo4j server. Every test exercises the service and API layers in isolation.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.api.entity_resolution import get_entity_resolution_service
from app.main import app
from app.repositories.graph_repository import GraphRepository
from app.schemas.entity_resolution import EntityCandidate, EntityResolutionResponse
from app.schemas.ner import NEREntity
from app.services.entity_resolution.entity_resolution_service import EntityResolutionService


# ---------------------------------------------------------------------------
# Test setup
# ---------------------------------------------------------------------------

os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_repository() -> MagicMock:
    repo = MagicMock(spec=GraphRepository)
    repo.find_entity_candidates.return_value = []
    return repo


@pytest.fixture()
def service(mock_repository: MagicMock) -> EntityResolutionService:
    return EntityResolutionService(repository=mock_repository)


@pytest.fixture()
def api_client(service: EntityResolutionService) -> Any:
    app.dependency_overrides[get_entity_resolution_service] = lambda: service
    yield client
    app.dependency_overrides.pop(get_entity_resolution_service, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ner_entity(text: str, label: str = "LOC", start: int = 0, end: int = 0) -> NEREntity:
    return NEREntity(text=text, label=label, start=start, end=end)


def _make_neo4j_node(node_id: str, name: str, label: str = "Unknown", aliases: list[str] | None = None) -> dict[str, Any]:
    node = MagicMock()
    node.labels = [label]
    props = {"id": node_id, "name": name}
    if aliases:
        props["aliases"] = aliases
    node.items.return_value = list(props.items())
    node.__getitem__ = lambda self, key: props[key]
    node.get = lambda self, key, default=None: props.get(key, default)
    return {"n": node}


# ---------------------------------------------------------------------------
# 1. Exact match
# ---------------------------------------------------------------------------


def test_exact_match_returns_matched(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = [
        _make_neo4j_node("country-001", "China", label="Country")
    ]

    result = service.resolve([_ner_entity("China")])

    assert result.resolved_entities[0].matched is True
    assert result.resolved_entities[0].node_id == "country-001"
    assert result.resolved_entities[0].node_name == "China"
    assert result.resolved_entities[0].match_method == "exact"
    assert result.resolved_entities[0].confidence == 1.0


# ---------------------------------------------------------------------------
# 2. Normalized exact match
# ---------------------------------------------------------------------------


def test_normalized_exact_match(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = [
        _make_neo4j_node("port-001", "Port of Rotterdam", label="Port")
    ]

    result = service.resolve([_ner_entity("  Port of Rotterdam ")])

    assert result.resolved_entities[0].matched is True
    assert result.resolved_entities[0].node_id == "port-001"
    assert result.resolved_entities[0].normalized_text == "port of rotterdam"
    assert result.resolved_entities[0].match_method == "exact"
    assert result.resolved_entities[0].confidence == 1.0


# ---------------------------------------------------------------------------
# 3. Fuzzy match
# ---------------------------------------------------------------------------


def test_fuzzy_match_returns_matched(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = [
        _make_neo4j_node("port-001", "Rotterdam Port", label="Port")
    ]

    result = service.resolve([_ner_entity("Rotterdam")])

    assert result.resolved_entities[0].matched is True
    assert result.resolved_entities[0].node_id == "port-001"
    assert result.resolved_entities[0].match_method == "fuzzy"
    assert result.resolved_entities[0].confidence >= 0.6


# ---------------------------------------------------------------------------
# 4. Weak fuzzy match rejected
# ---------------------------------------------------------------------------


def test_weak_fuzzy_match_rejected(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = [
        _make_neo4j_node("port-001", "Port of Rotterdam", label="Port")
    ]

    result = service.resolve([_ner_entity("Port")])

    assert result.resolved_entities[0].matched is False
    assert result.resolved_entities[0].match_method == "unresolved"


# ---------------------------------------------------------------------------
# 5. Unresolved entity
# ---------------------------------------------------------------------------


def test_unresolved_entity(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = []

    result = service.resolve([_ner_entity("Nonexistent Place")])

    assert result.resolved_entities[0].matched is False
    assert result.resolved_entities[0].match_method == "unresolved"
    assert result.resolved_entities[0].confidence == 0.0
    assert result.resolved_entities[0].node_id is None


# ---------------------------------------------------------------------------
# 6. Ambiguous entity (multiple exact matches)
# ---------------------------------------------------------------------------


def test_ambiguous_exact_match(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = [
        _make_neo4j_node("supplier-001", "ABC Corp", label="Supplier"),
        _make_neo4j_node("supplier-002", "ABC Corp", label="Supplier"),
    ]

    result = service.resolve([_ner_entity("ABC Corp")])

    assert result.resolved_entities[0].matched is False
    assert result.resolved_entities[0].match_method == "ambiguous"
    assert result.resolved_entities[0].candidates is not None
    assert len(result.resolved_entities[0].candidates) == 2


# ---------------------------------------------------------------------------
# 7. Empty candidate list
# ---------------------------------------------------------------------------


def test_empty_candidate_list(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = []

    result = service.resolve([_ner_entity("Unknown Entity")])

    assert result.resolved_entities[0].matched is False
    assert result.resolved_entities[0].match_method == "unresolved"


# ---------------------------------------------------------------------------
# 8. Repository failure (ServiceUnavailable)
# ---------------------------------------------------------------------------


def test_repository_service_unavailable(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.side_effect = ServiceUnavailable("Neo4j down")

    with pytest.raises(ServiceUnavailable):
        service.resolve([_ner_entity("China")])


# ---------------------------------------------------------------------------
# 9. Entity normalization
# ---------------------------------------------------------------------------


def test_entity_normalization(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = []

    result = service.resolve([_ner_entity("  Port of Rotterdam  ")])

    assert result.resolved_entities[0].original_text == "  Port of Rotterdam  "
    assert result.resolved_entities[0].normalized_text == "port of rotterdam"


# ---------------------------------------------------------------------------
# 10. Confidence scores
# ---------------------------------------------------------------------------


def test_exact_confidence_is_one(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = [
        _make_neo4j_node("country-001", "China", label="Country")
    ]

    result = service.resolve([_ner_entity("China")])

    assert result.resolved_entities[0].confidence == 1.0


def test_fuzzy_confidence_is_ratio(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = [
        _make_neo4j_node("port-001", "Rotterdam Port", label="Port")
    ]

    result = service.resolve([_ner_entity("Rotterdam")])

    assert 0.0 < result.resolved_entities[0].confidence < 1.0
    assert result.resolved_entities[0].match_method == "fuzzy"


# ---------------------------------------------------------------------------
# 11. API success
# ---------------------------------------------------------------------------


def test_resolve_endpoint_returns_200(api_client: Any) -> None:
    mock_repo = MagicMock(spec=GraphRepository)
    mock_repo.find_entity_candidates.return_value = [
        _make_neo4j_node("country-001", "China", label="Country")
    ]
    svc = EntityResolutionService(repository=mock_repo)
    app.dependency_overrides[get_entity_resolution_service] = lambda: svc
    try:
        payload = {
            "entities": [{"text": "China", "label": "LOC", "start": 0, "end": 5}],
        }
        response = api_client.post("/api/v1/news/resolve", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["resolved_entities"][0]["matched"] is True
        assert data["resolved_entities"][0]["node_id"] == "country-001"
    finally:
        app.dependency_overrides.pop(get_entity_resolution_service, None)


# ---------------------------------------------------------------------------
# 12. API validation
# ---------------------------------------------------------------------------


def test_resolve_endpoint_missing_entities_returns_422(api_client: Any) -> None:
    payload = {}
    response = api_client.post("/api/v1/news/resolve", json=payload)
    assert response.status_code == 422


def test_resolve_endpoint_empty_entities_returns_422(api_client: Any) -> None:
    payload = {"entities": []}
    response = api_client.post("/api/v1/news/resolve", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 13. API Neo4j unavailable
# ---------------------------------------------------------------------------


def test_resolve_endpoint_neo4j_unavailable_returns_503(api_client: Any) -> None:
    mock_repo = MagicMock(spec=GraphRepository)
    mock_repo.find_entity_candidates.side_effect = ServiceUnavailable("Neo4j down")
    svc = EntityResolutionService(repository=mock_repo)
    app.dependency_overrides[get_entity_resolution_service] = lambda: svc
    try:
        payload = {
            "entities": [{"text": "China", "label": "LOC", "start": 0, "end": 5}],
        }
        response = api_client.post("/api/v1/news/resolve", json=payload)
        assert response.status_code == 503
        assert "Neo4j down" not in str(response.json())
    finally:
        app.dependency_overrides.pop(get_entity_resolution_service, None)


# ---------------------------------------------------------------------------
# 14. Route registration
# ---------------------------------------------------------------------------


def test_resolve_route_is_registered() -> None:
    paths = {r.path for r in app.routes}
    assert "/api/v1/news/resolve" in paths


# ---------------------------------------------------------------------------
# 15. JSON serialization
# ---------------------------------------------------------------------------


def test_resolve_response_is_json_serializable(api_client: Any) -> None:
    mock_repo = MagicMock(spec=GraphRepository)
    mock_repo.find_entity_candidates.return_value = []
    svc = EntityResolutionService(repository=mock_repo)
    app.dependency_overrides[get_entity_resolution_service] = lambda: svc
    try:
        payload = {
            "entities": [{"text": "Nowhere", "label": "LOC", "start": 0, "end": 7}],
        }
        response = api_client.post("/api/v1/news/resolve", json=payload)
        assert response.status_code == 200
        response.json()  # must parse without error
    finally:
        app.dependency_overrides.pop(get_entity_resolution_service, None)


# ---------------------------------------------------------------------------
# 16. Article ID handling
# ---------------------------------------------------------------------------


def test_resolve_generates_article_id(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = []

    result = service.resolve([_ner_entity("Nowhere")])

    assert result.article_id is not None
    assert len(result.article_id) == 36


def test_resolve_uses_provided_article_id(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = []

    result = service.resolve([_ner_entity("Nowhere")], article_id="article-123")

    assert result.article_id == "article-123"


# ---------------------------------------------------------------------------
# 17. Counts
# ---------------------------------------------------------------------------


def test_resolve_counts(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.side_effect = [
        [_make_neo4j_node("country-001", "China", label="Country")],
        [],
        [_make_neo4j_node("supplier-001", "ABC Corp", label="Supplier"), _make_neo4j_node("supplier-002", "ABC Corp", label="Supplier")],
    ]

    result = service.resolve([
        _ner_entity("China"),
        _ner_entity("Nowhere"),
        _ner_entity("ABC Corp"),
    ])

    assert result.total_entities == 3
    assert result.resolved_count == 1
    assert result.unresolved_count == 1
    assert result.ambiguous_count == 1


# ---------------------------------------------------------------------------
# 18. Alias match
# ---------------------------------------------------------------------------


def test_alias_exact_match(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    node = MagicMock()
    node.labels = ["Port"]
    props = {"id": "port-001", "name": "Port of Rotterdam", "aliases": ["Rotterdam"]}
    node.items.return_value = list(props.items())
    node.__getitem__ = lambda self, key: props[key]
    node.get = lambda self, key, default=None: props.get(key, default)
    mock_repository.find_entity_candidates.return_value = [{"n": node}]

    result = service.resolve([_ner_entity("Rotterdam")])

    assert result.resolved_entities[0].matched is True
    assert result.resolved_entities[0].node_id == "port-001"
    assert result.resolved_entities[0].match_method == "exact"


# ---------------------------------------------------------------------------
# 19. Unresolved when Neo4j raises generic exception
# ---------------------------------------------------------------------------


def test_repository_generic_exception_raises(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.side_effect = RuntimeError("unexpected db error")

    with pytest.raises(Exception):
        service.resolve([_ner_entity("China")])


# ---------------------------------------------------------------------------
# 20. Confidence for unresolved is 0
# ---------------------------------------------------------------------------


def test_unresolved_confidence_is_zero(service: EntityResolutionService, mock_repository: MagicMock) -> None:
    mock_repository.find_entity_candidates.return_value = []

    result = service.resolve([_ner_entity("Unknown")])

    assert result.resolved_entities[0].confidence == 0.0
