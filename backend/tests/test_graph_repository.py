"""Unit tests for the graph repository layer (Module 4).

These tests mock the Neo4jDatabase so they do not require a running
Neo4j server. Every test exercises the repository in isolation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from neo4j import Record
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.repositories.graph_repository import GraphRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db() -> MagicMock:
    db = MagicMock()
    db.is_initialized = True
    return db


@pytest.fixture()
def repository(mock_db: MagicMock) -> GraphRepository:
    return GraphRepository(db=mock_db)


def _make_record(data: dict[str, Any]) -> MagicMock:
    record = MagicMock(spec=Record)
    record.items.return_value = list(data.items())
    return record


# ---------------------------------------------------------------------------
# 1. Successful read operation
# ---------------------------------------------------------------------------


def test_execute_read_returns_dict_list(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_read.return_value = [
        _make_record({"n": {"id": "1"}}),
        _make_record({"n": {"id": "2"}}),
    ]

    results = repository.execute_read("MATCH (n) RETURN n", {"limit": 2})

    assert len(results) == 2
    assert results[0] == {"n": {"id": "1"}}
    assert results[1] == {"n": {"id": "2"}}
    mock_db.execute_read.assert_called_once_with(
        "MATCH (n) RETURN n", {"limit": 2}
    )


# ---------------------------------------------------------------------------
# 2. Successful write operation
# ---------------------------------------------------------------------------


def test_execute_write_returns_dict_list(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_write.return_value = [_make_record({"n": {"id": "1"}})]

    results = repository.execute_write("CREATE (n) RETURN n", {})

    assert len(results) == 1
    assert results[0] == {"n": {"id": "1"}}
    mock_db.execute_write.assert_called_once_with("CREATE (n) RETURN n", {})


# ---------------------------------------------------------------------------
# 3. Node lookup
# ---------------------------------------------------------------------------


def test_get_node_by_id_found(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_read.return_value = [_make_record({"n": {"id": "abc"}})]

    result = repository.get_node_by_id("abc")

    assert result == {"n": {"id": "abc"}}
    mock_db.execute_read.assert_called_once()
    args, kwargs = mock_db.execute_read.call_args
    assert args[0] == "MATCH (n) WHERE n.id = $node_id RETURN n LIMIT 1"
    assert args[1] == {"node_id": "abc"}


def test_get_node_by_id_not_found(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_read.return_value = []

    result = repository.get_node_by_id("missing")

    assert result is None


# ---------------------------------------------------------------------------
# 4. Empty result handling
# ---------------------------------------------------------------------------


def test_get_nodes_empty(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_read.return_value = []

    results = repository.get_nodes(limit=10)

    assert results == []


def test_find_nodes_empty(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_read.return_value = []

    results = repository.find_nodes(label="Supplier", limit=10)

    assert results == []


# ---------------------------------------------------------------------------
# 5. Neo4j ServiceUnavailable handling
# ---------------------------------------------------------------------------


def test_execute_read_service_unavailable(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_read.side_effect = ServiceUnavailable("Driver not initialized")

    with pytest.raises(ServiceUnavailable):
        repository.execute_read("MATCH (n) RETURN n")


def test_get_node_by_id_service_unavailable(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_read.side_effect = ServiceUnavailable("Driver not initialized")

    with pytest.raises(ServiceUnavailable):
        repository.get_node_by_id("abc")


# ---------------------------------------------------------------------------
# 6. Query error handling
# ---------------------------------------------------------------------------


def test_execute_read_neo4j_error(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_read.side_effect = Neo4jError("Syntax error")

    with pytest.raises(Neo4jError):
        repository.execute_read("INVALID CYPHER")


def test_execute_write_neo4j_error(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_write.side_effect = Neo4jError("Constraint violation")

    with pytest.raises(Neo4jError):
        repository.execute_write("CREATE (n:Bad) RETURN n")


# ---------------------------------------------------------------------------
# 7. Parameter passing
# ---------------------------------------------------------------------------


def test_execute_read_passes_parameters(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_read.return_value = []

    repository.execute_read("MATCH (n) WHERE n.id = $id RETURN n", {"id": "123"})

    mock_db.execute_read.assert_called_once_with(
        "MATCH (n) WHERE n.id = $id RETURN n", {"id": "123"}
    )


def test_execute_write_passes_parameters(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_write.return_value = []

    repository.execute_write("CREATE (n {id: $id}) RETURN n", {"id": "456"})

    mock_db.execute_write.assert_called_once_with(
        "CREATE (n {id: $id}) RETURN n", {"id": "456"}
    )


# ---------------------------------------------------------------------------
# 8. No raw Neo4j driver objects exposed by the service
# ---------------------------------------------------------------------------


def test_execute_read_returns_plain_dicts(repository: GraphRepository, mock_db: MagicMock) -> None:
    mock_db.execute_read.return_value = [_make_record({"n": MagicMock()})]

    results = repository.execute_read("MATCH (n) RETURN n")

    for result in results:
        assert isinstance(result, dict)
        assert not hasattr(result, "session")
        assert not hasattr(result, "run")


# ---------------------------------------------------------------------------
# 9. Invalid input handling
# ---------------------------------------------------------------------------


def test_execute_read_rejects_empty_query(repository: GraphRepository) -> None:
    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        repository.execute_read("")


def test_execute_read_rejects_non_string_query(repository: GraphRepository) -> None:
    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        repository.execute_read(123)


def test_execute_write_rejects_empty_query(repository: GraphRepository) -> None:
    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        repository.execute_write("")


def test_get_node_by_id_rejects_empty_id(repository: GraphRepository) -> None:
    with pytest.raises(ValueError, match="node_id must be a non-empty string"):
        repository.get_node_by_id("")


def test_get_nodes_rejects_invalid_limit(repository: GraphRepository) -> None:
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        repository.get_nodes(limit=-1)


def test_find_neighbors_rejects_empty_node_id(repository: GraphRepository) -> None:
    with pytest.raises(ValueError, match="node_id must be a non-empty string"):
        repository.find_neighbors("")


# ---------------------------------------------------------------------------
# 10. Repository/service interaction (dependency injection)
# ---------------------------------------------------------------------------


def test_repository_can_be_injected_with_custom_db() -> None:
    custom_db = MagicMock()
    custom_db.execute_read.return_value = [_make_record({"n": {"id": "x"}})]

    repo = GraphRepository(db=custom_db)
    results = repo.get_nodes()

    assert len(results) == 1
    custom_db.execute_read.assert_called_once()
