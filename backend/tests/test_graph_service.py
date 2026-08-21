"""Unit tests for the graph service layer (Module 4).

These tests mock the GraphRepository so they do not require a running
Neo4j server.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.repositories.graph_repository import GraphRepository
from app.schemas.graph import (
    GraphNodeResponse,
    GraphQueryResponse,
    GraphRelationshipResponse,
    GraphResultResponse,
)
from app.services.graph_service import GraphService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_repository() -> MagicMock:
    return MagicMock(spec=GraphRepository)


@pytest.fixture()
def service(mock_repository: MagicMock) -> GraphService:
    return GraphService(repository=mock_repository)


# ---------------------------------------------------------------------------
# 1. Successful read operation
# ---------------------------------------------------------------------------


def test_get_nodes_returns_node_responses(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.get_nodes.return_value = [
        {"id": "1", "label": "Supplier", "properties": {"name": "A"}},
        {"id": "2", "label": "Factory", "properties": {"name": "B"}},
    ]

    results = service.get_nodes(limit=2)

    assert len(results) == 2
    assert isinstance(results[0], GraphNodeResponse)
    assert results[0].id == "1"
    assert results[0].label == "Supplier"
    mock_repository.get_nodes.assert_called_once_with(limit=2)


# ---------------------------------------------------------------------------
# 2. Successful write operation
# ---------------------------------------------------------------------------


def test_create_node_returns_node_response(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.execute_write.return_value = [
        {"id": "new-1", "label": "Supplier", "properties": {"name": "New Supplier"}}
    ]

    result = service.create_node(label="Supplier", properties={"name": "New Supplier"})

    assert isinstance(result, GraphNodeResponse)
    assert result.label == "Supplier"
    assert result.properties["name"] == "New Supplier"


# ---------------------------------------------------------------------------
# 3. Node lookup
# ---------------------------------------------------------------------------


def test_get_node_by_id_found(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.get_node_by_id.return_value = {"id": "abc", "label": "Factory", "properties": {}}

    result = service.get_node_by_id("abc")

    assert isinstance(result, GraphNodeResponse)
    assert result.id == "abc"
    assert result.label == "Factory"
    mock_repository.get_node_by_id.assert_called_once_with("abc")


def test_get_node_by_id_not_found(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.get_node_by_id.return_value = None

    result = service.get_node_by_id("missing")

    assert result is None


# ---------------------------------------------------------------------------
# 4. Empty result handling
# ---------------------------------------------------------------------------


def test_find_nodes_returns_empty_list(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.find_nodes.return_value = []

    results = service.find_nodes()

    assert results == []


def test_execute_read_query_empty_result(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.execute_read.return_value = []

    result = service.execute_read_query("MATCH (n) RETURN n")

    assert isinstance(result, GraphQueryResponse)
    assert result.count == 0
    assert result.results == []


# ---------------------------------------------------------------------------
# 5. Neo4j ServiceUnavailable handling
# ---------------------------------------------------------------------------


def test_get_node_by_id_service_unavailable(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.get_node_by_id.side_effect = ServiceUnavailable("Driver not initialized")

    with pytest.raises(ServiceUnavailable):
        service.get_node_by_id("abc")


def test_get_nodes_service_unavailable(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.get_nodes.side_effect = ServiceUnavailable("Driver not initialized")

    with pytest.raises(ServiceUnavailable):
        service.get_nodes()


# ---------------------------------------------------------------------------
# 6. Query error handling
# ---------------------------------------------------------------------------


def test_execute_read_query_neo4j_error(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.execute_read.side_effect = Neo4jError("Syntax error")

    with pytest.raises(Neo4jError):
        service.execute_read_query("INVALID CYPHER")


def test_execute_write_query_neo4j_error(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.execute_write.side_effect = Neo4jError("Constraint violation")

    with pytest.raises(Neo4jError):
        service.execute_write_query("CREATE (n:Bad) RETURN n")


# ---------------------------------------------------------------------------
# 7. Parameter passing
# ---------------------------------------------------------------------------


def test_execute_read_query_passes_parameters(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.execute_read.return_value = []

    service.execute_read_query("MATCH (n) WHERE n.id = $id RETURN n", {"id": "123"})

    mock_repository.execute_read.assert_called_once_with(
        "MATCH (n) WHERE n.id = $id RETURN n", {"id": "123"}
    )


# ---------------------------------------------------------------------------
# 8. No raw Neo4j driver objects exposed by the service
# ---------------------------------------------------------------------------


def test_service_returns_pydantic_models(service: GraphService, mock_repository: MagicMock) -> None:
    mock_repository.get_nodes.return_value = [
        {"id": "1", "label": "Supplier", "properties": {"name": "A"}},
    ]

    results = service.get_nodes()

    assert isinstance(results, list)
    assert isinstance(results[0], GraphNodeResponse)
    assert not hasattr(results[0], "session")
    assert not hasattr(results[0], "run")


# ---------------------------------------------------------------------------
# 9. Invalid input handling
# ---------------------------------------------------------------------------


def test_get_node_by_id_rejects_empty_id(service: GraphService) -> None:
    with pytest.raises(ValueError, match="node_id must be a non-empty string"):
        service.get_node_by_id("")


def test_get_nodes_rejects_invalid_limit(service: GraphService) -> None:
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        service.get_nodes(limit=-1)


def test_create_node_rejects_empty_label(service: GraphService) -> None:
    with pytest.raises(ValueError, match="label must be a non-empty string"):
        service.create_node(label="", properties={})


def test_create_node_rejects_non_dict_properties(service: GraphService) -> None:
    with pytest.raises(ValueError, match="properties must be a dictionary"):
        service.create_node(label="Supplier", properties="invalid")


def test_create_relationship_rejects_empty_source_id(service: GraphService) -> None:
    with pytest.raises(ValueError, match="source_id must be a non-empty string"):
        service.create_relationship("", "target", "REL")


def test_execute_read_query_rejects_empty_query(service: GraphService) -> None:
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        service.execute_read_query("")


# ---------------------------------------------------------------------------
# 10. Repository/service interaction
# ---------------------------------------------------------------------------


def test_service_uses_injected_repository() -> None:
    custom_repo = MagicMock(spec=GraphRepository)
    custom_repo.get_nodes.return_value = []
    service = GraphService(repository=custom_repo)

    service.get_nodes(limit=5)

    custom_repo.get_nodes.assert_called_once_with(limit=5)


def test_service_creates_default_repository_when_none() -> None:
    service = GraphService(repository=None)
    assert service._repository is not None
