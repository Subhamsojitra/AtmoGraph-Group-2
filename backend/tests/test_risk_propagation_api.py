"""API tests for the Module 10 risk propagation endpoint.

These tests inject a mocked :class:`RiskPropagationService` through
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

from app.api.risk_propagation import get_risk_propagation_service  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas.risk_propagation import RiskPropagationResponse  # noqa: E402
from app.services.risk.exceptions import EntityNotFoundError  # noqa: E402

client = TestClient(app)


@pytest.fixture()
def mock_service() -> MagicMock:
    """A RiskPropagationService mock used to isolate the API layer."""
    return MagicMock()


@pytest.fixture()
def api_client(mock_service: MagicMock) -> Any:
    """A base client with the propagation service dependency overridden."""
    app.dependency_overrides[get_risk_propagation_service] = lambda: mock_service
    yield client
    app.dependency_overrides.pop(get_risk_propagation_service, None)


def _response(**overrides: Any) -> RiskPropagationResponse:
    """Build a valid risk propagation response."""
    data = {
        "source_entity_id": "entity-001",
        "source_entity_name": "Port of Rotterdam",
        "source_risk_score": 80.0,
        "propagated": True,
        "affected_entities": [
            {
                "entity_id": "entity-002",
                "entity_name": "Gigafactory Assembly",
                "depth": 1,
                "propagated_risk_score": 40.0,
                "propagated_risk_level": "MEDIUM",
            }
        ],
        "affected_count": 1,
        "max_depth_reached": 1,
        "timestamp": datetime.now(timezone.utc),
    }
    data.update(overrides)
    return RiskPropagationResponse(**data)


def _payload(**overrides: Any) -> dict[str, Any]:
    """Build a valid request payload for the propagate endpoint."""
    data = {
        "entity_id": "entity-001",
        "entity_name": "Port of Rotterdam",
        "risk_score": 80.0,
        "max_depth": 2,
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- #
# 1. Successful request -> 200
# --------------------------------------------------------------------------- #


def test_successful_request_returns_200(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.propagate.return_value = _response()

    response = api_client.post("/api/v1/risk-propagation/propagate", json=_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["propagated"] is True
    assert payload["source_entity_id"] == "entity-001"
    assert payload["affected_count"] == 1
    assert payload["affected_entities"][0]["entity_id"] == "entity-002"
    mock_service.propagate.assert_called_once()


# --------------------------------------------------------------------------- #
# 2. Invalid request -> 422
# --------------------------------------------------------------------------- #


def test_invalid_high_score_returns_422(
    api_client: Any, mock_service: MagicMock
) -> None:
    response = api_client.post(
        "/api/v1/risk-propagation/propagate", json=_payload(risk_score=150.0)
    )

    assert response.status_code == 422
    mock_service.propagate.assert_not_called()


def test_invalid_low_score_returns_422(
    api_client: Any, mock_service: MagicMock
) -> None:
    response = api_client.post(
        "/api/v1/risk-propagation/propagate", json=_payload(risk_score=-5.0)
    )

    assert response.status_code == 422


def test_missing_risk_score_returns_422(
    api_client: Any, mock_service: MagicMock
) -> None:
    response = api_client.post(
        "/api/v1/risk-propagation/propagate", json={"entity_id": "entity-001"}
    )

    assert response.status_code == 422


def test_invalid_max_depth_returns_422(
    api_client: Any, mock_service: MagicMock
) -> None:
    response = api_client.post(
        "/api/v1/risk-propagation/propagate", json=_payload(max_depth=0)
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 3. Unresolved entity -> controlled 200 with propagated=False
# --------------------------------------------------------------------------- #


def test_unresolved_entity_returns_200_propagated_false(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.propagate.return_value = _response(
        propagated=False,
        source_entity_id="",
        affected_entities=[],
        affected_count=0,
        error="Entity is unresolved (no node_id provided); no propagation was run",
    )

    response = api_client.post(
        "/api/v1/risk-propagation/propagate", json=_payload(entity_id=None)
    )

    assert response.status_code == 200
    assert response.json()["propagated"] is False
    mock_service.propagate.assert_called_once()


# --------------------------------------------------------------------------- #
# 4. Source entity not found -> 404
# --------------------------------------------------------------------------- #


def test_entity_not_found_returns_404(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.propagate.side_effect = EntityNotFoundError("not found")

    response = api_client.post("/api/v1/risk-propagation/propagate", json=_payload())

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# 5. Neo4j unavailable -> 503
# --------------------------------------------------------------------------- #


def test_neo4j_unavailable_returns_503(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.propagate.side_effect = ServiceUnavailable("Neo4j down")

    response = api_client.post("/api/v1/risk-propagation/propagate", json=_payload())

    assert response.status_code == 503
    # Internal details must not leak to the client.
    assert "Neo4j down" not in response.text


# --------------------------------------------------------------------------- #
# 6. Response is JSON serializable
# --------------------------------------------------------------------------- #


def test_response_is_json(api_client: Any, mock_service: MagicMock) -> None:
    import json

    mock_service.propagate.return_value = _response()

    response = api_client.post("/api/v1/risk-propagation/propagate", json=_payload())

    assert response.status_code == 200
    payload = response.json()  # must parse without error
    json.dumps(payload)  # must re-serialize without error
    assert isinstance(payload["propagated"], bool)


# --------------------------------------------------------------------------- #
# 7/8. Route + OpenAPI registration
# --------------------------------------------------------------------------- #


def test_propagation_route_is_registered() -> None:
    paths = {r.path for r in app.routes}
    assert "/api/v1/risk-propagation/propagate" in paths


def test_propagation_endpoint_in_openapi() -> None:
    openapi = app.openapi()
    assert "/api/v1/risk-propagation/propagate" in openapi["paths"]
    assert "post" in openapi["paths"]["/api/v1/risk-propagation/propagate"]