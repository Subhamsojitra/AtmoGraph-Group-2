"""API tests for the Module 9 risk state update endpoint.

These tests inject a mocked :class:`RiskService` through
``app.dependency_overrides`` so they do NOT require a running Neo4j server. They
verify the route exists, is registered, appears in OpenAPI, and maps domain
errors to the correct HTTP status codes.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

# Fake Neo4j env vars must be set before importing the app (pydantic-settings).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

from app.api.news import get_risk_service  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas.risk import RiskStateUpdateResponse  # noqa: E402
from app.services.risk.exceptions import EntityNotFoundError  # noqa: E402

client = TestClient(app)


@pytest.fixture()
def mock_service() -> MagicMock:
    """A RiskService mock used to isolate the API layer from Neo4j."""
    return MagicMock()


@pytest.fixture()
def api_client(mock_service: MagicMock) -> Any:
    """A base client with the risk service dependency overridden."""
    app.dependency_overrides[get_risk_service] = lambda: mock_service
    yield client
    app.dependency_overrides.pop(get_risk_service, None)


def _response(**overrides: Any) -> RiskStateUpdateResponse:
    """Build a valid risk update response."""
    data = {
        "entity_id": "entity-001",
        "entity_name": "Port of Rotterdam",
        "updated": True,
        "previous_risk_score": 30.0,
        "new_risk_score": 85.0,
        "previous_risk_level": "LOW",
        "new_risk_level": "HIGH",
        "reason": "Port disruption",
        "timestamp": datetime.now(timezone.utc),
    }
    data.update(overrides)
    return RiskStateUpdateResponse(**data)


def _payload(**overrides: Any) -> dict[str, Any]:
    """Build a valid request payload for the risk-update endpoint."""
    data = {"entity_id": "entity-001", "risk_score": 85.0, "reason": "Port disruption"}
    data.update(overrides)
    return data
# --------------------------------------------------------------------------- #
# 1. POST endpoint exists & 2. successful request
# --------------------------------------------------------------------------- #


def test_successful_request_returns_200(api_client: Any, mock_service: MagicMock) -> None:
    mock_service.update_risk_state.return_value = _response()

    response = api_client.post("/api/v1/news/risk-update", json=_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["updated"] is True
    assert payload["entity_id"] == "entity-001"
    assert payload["new_risk_score"] == 85.0
    assert payload["new_risk_level"] == "HIGH"
    assert payload["previous_risk_level"] == "LOW"
    mock_service.update_risk_state.assert_called_once()


# --------------------------------------------------------------------------- #
# 3/4. Invalid request -> 422
# --------------------------------------------------------------------------- #


def test_invalid_high_score_returns_422(api_client: Any, mock_service: MagicMock) -> None:
    response = api_client.post(
        "/api/v1/news/risk-update", json=_payload(risk_score=150.0)
    )

    assert response.status_code == 422
    mock_service.update_risk_state.assert_not_called()


def test_invalid_low_score_returns_422(api_client: Any, mock_service: MagicMock) -> None:
    response = api_client.post(
        "/api/v1/news/risk-update", json=_payload(risk_score=-5.0)
    )

    assert response.status_code == 422


def test_missing_risk_score_returns_422(api_client: Any, mock_service: MagicMock) -> None:
    response = api_client.post(
        "/api/v1/news/risk-update", json={"entity_id": "entity-001"}
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 5. Unresolved entity -> controlled 200 with updated=False
# --------------------------------------------------------------------------- #


def test_unresolved_entity_returns_200_updated_false(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.update_risk_state.return_value = _response(
        updated=False,
        entity_id="",
        new_risk_level="HIGH",
        error="Entity is unresolved (no node_id provided); no graph node was updated",
    )

    # Entity omitted from the payload => unresolved.
    response = api_client.post(
        "/api/v1/news/risk-update", json={"risk_score": 85.0}
    )

    assert response.status_code == 200
    assert response.json()["updated"] is False
    mock_service.update_risk_state.assert_called_once()


# --------------------------------------------------------------------------- #
# 6. Entity not found -> 404
# --------------------------------------------------------------------------- #


def test_entity_not_found_returns_404(api_client: Any, mock_service: MagicMock) -> None:
    mock_service.update_risk_state.side_effect = EntityNotFoundError("not found")

    response = api_client.post("/api/v1/news/risk-update", json=_payload())

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# 7. Neo4j unavailable -> 503
# --------------------------------------------------------------------------- #


def test_neo4j_unavailable_returns_503(api_client: Any, mock_service: MagicMock) -> None:
    mock_service.update_risk_state.side_effect = ServiceUnavailable("Neo4j down")

    response = api_client.post("/api/v1/news/risk-update", json=_payload())

    assert response.status_code == 503
    # Internal details must not leak to the client.
    assert "Neo4j down" not in response.text


# --------------------------------------------------------------------------- #
# 8. Response is JSON serializable
# --------------------------------------------------------------------------- #


def test_response_is_json(api_client: Any, mock_service: MagicMock) -> None:
    import json

    mock_service.update_risk_state.return_value = _response()

    response = api_client.post("/api/v1/news/risk-update", json=_payload())

    assert response.status_code == 200
    payload = response.json()  # must parse without error
    json.dumps(payload)  # must re-serialize without error
    assert isinstance(payload["updated"], bool)


# --------------------------------------------------------------------------- #
# 9/10. Route + OpenAPI registration
# --------------------------------------------------------------------------- #


def test_risk_route_is_registered() -> None:
    paths = {r.path for r in app.routes}
    assert "/api/v1/news/risk-update" in paths


def test_risk_endpoint_in_openapi() -> None:
    openapi = app.openapi()
    assert "/api/v1/news/risk-update" in openapi["paths"]
    assert "post" in openapi["paths"]["/api/v1/news/risk-update"]