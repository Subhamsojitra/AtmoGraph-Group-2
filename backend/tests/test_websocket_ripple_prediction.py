"""Tests for Module 17 Part 4: WebSocket -> RipplePredictionService integration.

These tests mount the real ``/api/v1/ws`` route but MOCK the ripple prediction
service at the WebSocket dependency, so NONE of them require a running Neo4j
server, a trained GNN checkpoint or a GPU. They verify the Module 17 request
dispatch, response normalization, error mapping and client isolation around
the EXISTING Module 17 service contract.

Conventions follow ``tests/test_websocket.py`` (Module 15),
``tests/test_websocket_prediction.py`` (Module 16) and
``tests/test_ripple_prediction_service.py`` (Module 17 Part 3).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Fake Neo4j env vars must be set before importing the app (pydantic-settings).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

from app.main import app  # noqa: E402
from app.schemas.websocket import (  # noqa: E402
    ERROR_INVALID_MESSAGE,
    ERROR_PREDICTION_FAILED,
    MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT,
)
from app.services.ripple_prediction.exceptions import RipplePredictionError
from app.services.ripple_prediction.result_schema import (
    RippleAffectedEntity,
    RipplePredictionResult,
)
from app.services.risk.exceptions import EntityNotFoundError
from app.services.websocket_manager import reset_connection_manager  # noqa: E402

client = TestClient(app)
WS_URL = "/api/v1/ws"


@pytest.fixture(autouse=True)
def _clean_connection_manager() -> None:
    """Start every test with a fresh connection manager registry."""
    reset_connection_manager()
    yield
    reset_connection_manager()


@pytest.fixture()
def mock_ripple_service() -> MagicMock:
    """A RipplePredictionService mock used to isolate the WebSocket layer."""
    return MagicMock()


@pytest.fixture()
def api_client(mock_ripple_service: MagicMock) -> Any:
    """A base WebSocket client with the ripple service dependency overridden."""
    from app.api.websocket import get_ripple_prediction_service

    app.dependency_overrides[get_ripple_prediction_service] = (
        lambda: mock_ripple_service
    )
    yield client
    app.dependency_overrides.pop(get_ripple_prediction_service, None)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_ripple_result(
    source_entity_id: str = "SRC_001",
    source_entity_name: str = "Source Entity",
    source_risk_score: float = 75.0,
    affected_entities: list[RippleAffectedEntity] | None = None,
    propagated: bool = True,
    error: str | None = None,
) -> RipplePredictionResult:
    """Build a RipplePredictionResult with sensible defaults."""
    affected = affected_entities or [
        RippleAffectedEntity(
            entity_id="DST_001",
            entity_name="Downstream A",
            depth=1,
            propagated_risk_score=50.0,
            propagated_risk_level="MEDIUM",
            gnn_prediction=0.75,
        ),
    ]
    return RipplePredictionResult(
        source_entity_id=source_entity_id,
        source_entity_name=source_entity_name,
        source_risk_score=source_risk_score,
        propagated=propagated,
        affected_entities=affected,
        affected_count=len(affected),
        max_depth_reached=1,
        prediction_count=1,
        error=error,
    )


# --------------------------------------------------------------------------- #
# 1. Valid ripple_prediction request
# --------------------------------------------------------------------------- #


def test_valid_ripple_prediction_returns_result(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """A valid ripple_prediction request reaches the service and returns a result."""
    mock_ripple_service.predict.return_value = make_ripple_result()

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "ripple_prediction", "data": {"entity_id": "SRC_001"}}
        )
        response = websocket.receive_json()

        assert response["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT
        assert response["data"]["source_entity_id"] == "SRC_001"
        assert response["data"]["affected_count"] == 1
        assert response["data"]["propagated"] is True
        mock_ripple_service.predict.assert_called_once()


def test_ripple_prediction_service_receives_validated_request(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """The service receives a validated request with the correct entity_id."""
    mock_ripple_service.predict.return_value = make_ripple_result()

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {
                "type": "ripple_prediction",
                "data": {"entity_id": "ENTITY-42", "risk_score": 80.0},
            }
        )
        websocket.receive_json()  # consume result

        call_args = mock_ripple_service.predict.call_args[0][0]
        assert call_args.entity_id == "ENTITY-42"
        assert call_args.risk_score == 80.0


# --------------------------------------------------------------------------- #
# 2. Validation errors (missing / empty entity_id)
# --------------------------------------------------------------------------- #


def test_ripple_prediction_missing_entity_id_returns_invalid_message(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """A ripple_prediction with no data payload is rejected."""
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "ripple_prediction"})
        error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_INVALID_MESSAGE
        mock_ripple_service.predict.assert_not_called()


def test_ripple_prediction_empty_entity_id_returns_invalid_message(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """A blank/whitespace-only entity_id is rejected by the schema."""
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "ripple_prediction", "data": {"entity_id": "   "}}
        )
        error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_INVALID_MESSAGE
        mock_ripple_service.predict.assert_not_called()


# --------------------------------------------------------------------------- #
# 3. Service error handling
# --------------------------------------------------------------------------- #


def test_ripple_prediction_entity_not_found_returns_structured_error(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """An unknown entity is translated to a structured error."""
    mock_ripple_service.predict.side_effect = EntityNotFoundError("not found")

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "ripple_prediction", "data": {"entity_id": "UNKNOWN"}}
        )
        error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_INVALID_MESSAGE
        assert "stack" not in error["error"]["message"].lower()


def test_ripple_prediction_service_failure_returns_structured_error(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """A RipplePredictionError is translated to a structured error."""
    mock_ripple_service.predict.side_effect = RipplePredictionError("boom")

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "ripple_prediction", "data": {"entity_id": "SRC_001"}}
        )
        error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_PREDICTION_FAILED
        assert "boom" not in error["error"]["message"]


def test_ripple_prediction_unexpected_exception_returns_structured_error(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """An unexpected exception is caught and translated to a structured error."""
    mock_ripple_service.predict.side_effect = RuntimeError("unexpected")

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "ripple_prediction", "data": {"entity_id": "SRC_001"}}
        )
        error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_PREDICTION_FAILED
        assert "unexpected" not in error["error"]["message"]


# --------------------------------------------------------------------------- #
# 4. Connection stability
# --------------------------------------------------------------------------- #


def test_ripple_prediction_failure_keeps_connection_alive(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """A failing ripple_prediction does NOT terminate the connection."""
    mock_ripple_service.predict.side_effect = RipplePredictionError("boom")

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "ripple_prediction", "data": {"entity_id": "SRC_001"}}
        )
        error = websocket.receive_json()
        assert error["type"] == "error"

        # Connection is still alive: ping/pong works
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


# --------------------------------------------------------------------------- #
# 5. Backward compatibility
# --------------------------------------------------------------------------- #


def test_ping_pong_still_works_around_ripple_predictions(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """Module 15 ping/pong remains intact before and after ripple predictions."""
    mock_ripple_service.predict.return_value = make_ripple_result()

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"
        websocket.send_json(
            {"type": "ripple_prediction", "data": {"entity_id": "SRC_001"}}
        )
        assert websocket.receive_json()["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_multiple_clients_remain_isolated_for_ripple_predictions(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """A failing request for one client never affects another client."""
    def failing_once(*args: Any, **kwargs: Any) -> RipplePredictionResult:
        if mock_ripple_service.predict.call_count == 1:
            raise RipplePredictionError("boom")
        return make_ripple_result()

    mock_ripple_service.predict.side_effect = failing_once

    with api_client.websocket_connect(WS_URL) as ws1:
        ws1.receive_json()  # consume connected
        with api_client.websocket_connect(WS_URL) as ws2:
            ws2.receive_json()  # consume connected

            # Client 1 sees its own failure...
            ws1.send_json(
                {"type": "ripple_prediction", "data": {"entity_id": "SRC_001"}}
            )
            error = ws1.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == ERROR_PREDICTION_FAILED

            # ...while client 2 remains fully functional...
            ws2.send_json({"type": "ping"})
            assert ws2.receive_json()["type"] == "pong"
            ws2.send_json(
                {"type": "ripple_prediction", "data": {"entity_id": "SRC_001"}}
            )
            result = ws2.receive_json()
            assert result["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT

            # ...and client 1 is still usable after its failed request.
            ws1.send_json({"type": "ping"})
            assert ws1.receive_json()["type"] == "pong"
