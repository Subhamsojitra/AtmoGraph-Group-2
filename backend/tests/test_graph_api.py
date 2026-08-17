"""Unit tests for the Graph API layer (Module 5).

These tests inject a mocked :class:`GraphService` through
``app.dependency_overrides`` so they do NOT require a running Neo4j server.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock

from neo4j.exceptions import ServiceUnavailable
import pytest
from fastapi.testclient import TestClient

# Set fake Neo4j environment variables before importing the app (required by
# the pydantic-settings configuration, mirrors test_health.py).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

from app.api.graph import get_graph_service
from app.main import app
from app.schemas.graph import GraphNodeResponse

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_service() -> MagicMock:
    """A GraphService mock used to isolate the API layer from Neo4j."""
    return MagicMock()


@pytest.fixture()
def api_client(mock_service: MagicMock) -> Any:
    """A base client with the graph service dependency overridden."""
    app.dependency_overrides[get_graph_service] = lambda: mock_service
    yield client
    app.dependency_overrides.pop(get_graph_service, None)


def _node(node_id: str = "1", label: str = "Supplier") -> GraphNodeResponse:
    return GraphNodeResponse(id=node_id, label=label, properties={"name": f"Node {node_id}"})


# ---------------------------------------------------------------------------
# 1. GET node by ID -> 200
# ---------------------------------------------------------------------------


def test_get_node_by_id_returns_200(api_client: Any, mock_service: MagicMock) -> None:
    mock_service.get_node_by_id.return_value = _node("abc", "Factory")

    response = api_client.get("/api/v1/graph/nodes/abc")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "abc"
    assert payload["label"] == "Factory"
    mock_service.get_node_by_id.assert_called_once_with("abc")


# ---------------------------------------------------------------------------
# 2. GET node by ID -> 404
# ---------------------------------------------------------------------------


def test_get_node_by_id_returns_404_when_missing(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_node_by_id.return_value = None

    response = api_client.get("/api/v1/graph/nodes/missing")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 3. GET nodes -> 200
# ---------------------------------------------------------------------------


def test_get_nodes_returns_200(api_client: Any, mock_service: MagicMock) -> None:
    mock_service.get_nodes.return_value = [_node("1"), _node("2", "Factory")]

    response = api_client.get("/api/v1/graph/nodes")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["id"] == "1"
    mock_service.get_nodes.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Empty node list -> 200
# ---------------------------------------------------------------------------


def test_get_nodes_empty_returns_200(api_client: Any, mock_service: MagicMock) -> None:
    mock_service.get_nodes.return_value = []

    response = api_client.get("/api/v1/graph/nodes")

    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# 5. Invalid limit -> 422
# ---------------------------------------------------------------------------


def test_get_nodes_invalid_limit_returns_422(
    api_client: Any, mock_service: MagicMock
) -> None:
    response = api_client.get("/api/v1/graph/nodes?limit=0")

    assert response.status_code == 422
    mock_service.get_nodes.assert_not_called()
# ---------------------------------------------------------------------------
# 6. Search -> 200
# ---------------------------------------------------------------------------


def test_search_returns_200(api_client: Any, mock_service: MagicMock) -> None:
    mock_service.find_nodes.return_value = [_node("1", "Supplier")]

    response = api_client.get(
        "/api/v1/graph/search?label=Supplier&properties=%7B%22name%22%3A%22A%22%7D"
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload[0]["label"] == "Supplier"
    called_kwargs = mock_service.find_nodes.call_args.kwargs
    assert called_kwargs["label"] == "Supplier"
    assert called_kwargs["properties"] == {"name": "A"}


# ---------------------------------------------------------------------------
# 7. Neighbors -> 200
# ---------------------------------------------------------------------------


def test_neighbors_returns_200(api_client: Any, mock_service: MagicMock) -> None:
    mock_service.find_neighbors.return_value = [_node("2", "Factory")]

    response = api_client.get("/api/v1/graph/nodes/1/neighbors")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["id"] == "2"
    mock_service.find_neighbors.assert_called_once()
    assert mock_service.find_neighbors.call_args.kwargs["node_id"] == "1"


# ---------------------------------------------------------------------------
# 8. Neo4j unavailable -> 503
# ---------------------------------------------------------------------------


def test_graph_returns_503_when_neo4j_unavailable(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_node_by_id.side_effect = ServiceUnavailable("Neo4j down")

    response = api_client.get("/api/v1/graph/nodes/abc")

    assert response.status_code == 503
    # Internal details / error text must not leak to the client.
    assert "Neo4j down" not in json.dumps(response.json())


# ---------------------------------------------------------------------------
# 9. Invalid input -> 4xx
# ---------------------------------------------------------------------------


def test_search_invalid_properties_returns_422(
    api_client: Any, mock_service: MagicMock
) -> None:
    response = api_client.get("/api/v1/graph/search?properties=%7Bnot-json%7D")

    assert response.status_code == 422
    mock_service.find_nodes.assert_not_called()


def test_search_non_object_properties_returns_422(
    api_client: Any, mock_service: MagicMock
) -> None:
    response = api_client.get("/api/v1/graph/search?properties=%5B1%2C2%5D")

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 10. Response is JSON serializable
# ---------------------------------------------------------------------------


def test_response_is_json_serializable(api_client: Any, mock_service: MagicMock) -> None:
    mock_service.get_node_by_id.return_value = _node("1", "Supplier")

    response = api_client.get("/api/v1/graph/nodes/1")

    assert response.status_code == 200
    payload = response.json()  # must parse without error
    json.dumps(payload)  # must re-serialize without error
    assert payload["id"] == "1"
    assert isinstance(payload["properties"], dict)


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


def test_graph_routes_are_registered() -> None:
    paths = {r.path for r in app.routes}
    assert "/api/v1/graph/nodes/{node_id}" in paths
    assert "/api/v1/graph/nodes" in paths
    assert "/api/v1/graph/search" in paths
    assert "/api/v1/graph/nodes/{node_id}/neighbors" in paths