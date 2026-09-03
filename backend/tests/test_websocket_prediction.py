"""Tests for Module 16: WebSocket -> PredictionService -> GNN integration.

These tests mount the real ``/api/v1/ws`` route but MOCK the prediction
service at the WebSocket dependency, so NONE of them require a running Neo4j
server, a trained GNN checkpoint or a GPU. They verify the Module 16 request
dispatch, response normalization, node selection, error mapping and client
isolation around the EXISTING Module 14 service contract (which is itself
exercised by ``tests/test_prediction_service.py`` and
``tests/test_gnn_prediction.py``).

Conventions follow ``tests/test_websocket.py`` (Module 15) and
``tests/test_prediction_api.py`` (Module 14, mocked service).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Fake Neo4j env vars must be set before importing the app (pydantic-settings).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

from app.main import app  # noqa: E402
from app.ml.exceptions import (  # noqa: E402
    EmptyGraphError,
    GNNModelNotAvailableError,
    GNNPredictionRuntimeError,
)
from app.schemas.prediction import NodePrediction, PredictionResponse  # noqa: E402
from app.schemas.websocket import (  # noqa: E402
    ERROR_INVALID_JSON,
    ERROR_INVALID_MESSAGE,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_NODE_NOT_FOUND,
    ERROR_PREDICTION_FAILED,
    MESSAGE_TYPE_PREDICTION_REQUEST,
    MESSAGE_TYPE_PREDICTION_RESULT,
    WebSocketPredictionRequest,
    build_prediction_result_message,
)
from app.services.prediction_service import ModelNotAvailableError  # noqa: E402
from app.services.websocket_prediction import (  # noqa: E402
    get_websocket_prediction_service,
)
from app.services.websocket_manager import reset_connection_manager  # noqa: E402

client = TestClient(app)
WS_URL = "/api/v1/ws"


def make_response(*pairs: tuple[str, float]) -> PredictionResponse:
    """Build a valid Module 14 PredictionResponse from (node_id, value) pairs."""
    predictions = [
        NodePrediction(node_id=node_id, prediction=value)
        for node_id, value in pairs
    ]
    return PredictionResponse(
        predictions=predictions, prediction_count=len(predictions)
    )


@pytest.fixture(autouse=True)
def _clean_connection_manager() -> None:
    """Start every test with a fresh connection manager registry."""
    reset_connection_manager()
    yield
    reset_connection_manager()


@pytest.fixture()
def mock_service() -> MagicMock:
    """A PredictionService mock used to isolate the WebSocket dispatch layer."""
    return MagicMock()


@pytest.fixture()
def api_client(mock_service: MagicMock) -> Any:
    """A base WebSocket client with the prediction service dependency overridden."""
    app.dependency_overrides[get_websocket_prediction_service] = lambda: mock_service
    yield client
    app.dependency_overrides.pop(get_websocket_prediction_service, None)


# --------------------------------------------------------------------------- #
# 0. Schema / payload contracts
# --------------------------------------------------------------------------- #


def test_prediction_request_schema_accepts_empty_data() -> None:
    """"No data" means "predict for the whole graph" (REST-equivalent request)."""
    request = WebSocketPredictionRequest()
    assert request.node_id is None
    assert request.relationship_types is None


def test_prediction_request_schema_trims_node_id() -> None:
    request = WebSocketPredictionRequest(node_id="  supplier-001  ")
    assert request.node_id == "supplier-001"


def test_prediction_request_schema_rejects_blank_node_id() -> None:
    with pytest.raises(ValidationError):
        WebSocketPredictionRequest(node_id="   ")


def test_prediction_request_schema_rejects_non_string_node_id() -> None:
    with pytest.raises(ValidationError):
        WebSocketPredictionRequest(node_id=123)


def test_prediction_request_schema_rejects_unknown_fields() -> None:
    """The frontend-style camelCase ``nodeId`` is never silently accepted."""
    with pytest.raises(ValidationError):
        WebSocketPredictionRequest(nodeId="supplier-001")


def test_prediction_result_envelope_is_json_safe() -> None:
    """The prediction_result envelope reuses the Module 14 response shape."""
    envelope = build_prediction_result_message(
        make_response(("supplier-001", 72.4), ("supplier-002", 5.1))
    )
    assert set(envelope.keys()) == {"type", "timestamp", "data"}
    assert envelope["type"] == MESSAGE_TYPE_PREDICTION_RESULT
    data = envelope["data"]
    assert data["prediction_count"] == 2
    assert data["predictions"] == [
        {"node_id": "supplier-001", "prediction": 72.4},
        {"node_id": "supplier-002", "prediction": 5.1},
    ]
    assert "requested_node_id" not in data


def test_prediction_result_envelope_echoes_requested_node() -> None:
    envelope = build_prediction_result_message(
        make_response(("supplier-001", 72.4)),
        requested_node_id="supplier-001",
    )
    data = envelope["data"]
    assert data["requested_node_id"] == "supplier-001"
    assert data["prediction_count"] == 1


# --------------------------------------------------------------------------- #
# 1. Request dispatch & success (mocked PredictionService)
# --------------------------------------------------------------------------- #


def test_prediction_request_is_accepted_and_returns_result(
    api_client: Any, mock_service: MagicMock
) -> None:
    """A whole-graph prediction_request returns a prediction_result envelope."""
    mock_service.get_predictions.return_value = make_response(
        ("supplier-001", 72.4), ("supplier-002", 5.1)
    )
    with api_client.websocket_connect(WS_URL) as websocket:
        assert websocket.receive_json()["type"] == "connected"
        websocket.send_json({"type": "prediction_request"})
        result = websocket.receive_json()
        assert result["type"] == MESSAGE_TYPE_PREDICTION_RESULT
        data = result["data"]
        assert data["prediction_count"] == 2
        assert data["predictions"][0] == {
            "node_id": "supplier-001",
            "prediction": 72.4,
        }
        assert data["predictions"][1] == {
            "node_id": "supplier-002",
            "prediction": 5.1,
        }
        mock_service.get_predictions.assert_called_once()


def test_prediction_request_calls_prediction_service_with_payload(
    api_client: Any, mock_service: MagicMock
) -> None:
    """relationship_types is passed through to the Module 14 service."""
    mock_service.get_predictions.return_value = make_response(("n0", 1.0))
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {
                "type": "prediction_request",
                "data": {"relationship_types": ["SUPPLIES"]},
            }
        )
        assert websocket.receive_json()["type"] == MESSAGE_TYPE_PREDICTION_RESULT
        args = mock_service.get_predictions.call_args.args
        assert len(args) == 1
        request = args[0]
        assert isinstance(request, WebSocketPredictionRequest)
        assert request.relationship_types == ["SUPPLIES"]
        assert request.node_id is None


def test_single_node_request_returns_only_that_node(
    api_client: Any, mock_service: MagicMock
) -> None:
    """node_id is preserved: only the requested node's REAL output is returned."""
    mock_service.get_predictions.return_value = make_response(
        ("supplier-001", 72.4), ("supplier-002", 5.1)
    )
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {
                "type": "prediction_request",
                "data": {"node_id": "supplier-001"},
            }
        )
        result = websocket.receive_json()
        assert result["type"] == MESSAGE_TYPE_PREDICTION_RESULT
        data = result["data"]
        assert data["requested_node_id"] == "supplier-001"
        assert data["prediction_count"] == 1
        assert data["predictions"] == [
            {"node_id": "supplier-001", "prediction": 72.4}
        ]


def test_connected_message_advertises_prediction_request(
    api_client: Any, mock_service: MagicMock
) -> None:
    """The transport advertises prediction_request as a supported message."""
    with api_client.websocket_connect(WS_URL) as websocket:
        message = websocket.receive_json()
        supported = message["data"]["supported_client_messages"]
        assert "ping" in supported
        assert MESSAGE_TYPE_PREDICTION_REQUEST in supported


# --------------------------------------------------------------------------- #
# 2. Invalid requests (structured errors; the connection stays usable)
# --------------------------------------------------------------------------- #


def test_malformed_prediction_request_json_returns_structured_error(
    api_client: Any, mock_service: MagicMock
) -> None:
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_text("{not json")
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_INVALID_JSON
        # Connection remains usable, and no inference ran.
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"
        mock_service.get_predictions.assert_not_called()


def test_prediction_request_with_wrong_field_type_returns_structured_error(
    api_client: Any, mock_service: MagicMock
) -> None:
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "prediction_request", "data": {"node_id": 123}}
        )
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_INVALID_MESSAGE
        assert "node_id" in error["error"]["message"]
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"
        mock_service.get_predictions.assert_not_called()


def test_prediction_request_with_unknown_field_returns_structured_error(
    api_client: Any, mock_service: MagicMock
) -> None:
    """The frontend-style ``nodeId`` is rejected, never silently ignored."""
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "prediction_request", "data": {"nodeId": "supplier-001"}}
        )
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_INVALID_MESSAGE
        assert "nodeId" in error["error"]["message"]
        mock_service.get_predictions.assert_not_called()


def test_invalid_node_id_returns_structured_error(
    api_client: Any, mock_service: MagicMock
) -> None:
    """A node id absent from the graph yields NODE_NOT_FOUND, safely."""
    mock_service.get_predictions.return_value = make_response(("n0", 1.0))
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "prediction_request", "data": {"node_id": "ghost-node"}}
        )
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_NODE_NOT_FOUND
        assert "ghost-node" in error["error"]["message"]
        # The connection is still usable after the failed request.
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


# --------------------------------------------------------------------------- #
# 3. Service / model failures (structured errors, no crashes, no leaks)
# --------------------------------------------------------------------------- #


def test_model_not_configured_returns_structured_error(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = ModelNotAvailableError(
        "no trained GNN checkpoint is configured"
    )
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "prediction_request"})
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_MODEL_UNAVAILABLE
        assert "trained" in error["error"]["message"].lower()
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_missing_checkpoint_file_returns_structured_error(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = GNNModelNotAvailableError(
        "trained GNN checkpoint not found: C:/secret/models/model.pt"
    )
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "prediction_request"})
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_MODEL_UNAVAILABLE
        # No filesystem paths leak to the client.
        assert "C:" not in error["error"]["message"]
        assert ".pt" not in error["error"]["message"]


def test_inference_failure_does_not_break_connection(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = GNNPredictionRuntimeError("boom")
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "prediction_request"})
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_PREDICTION_FAILED
        assert "boom" not in error["error"]["message"]
        assert "Traceback" not in error["error"]["message"]
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_empty_graph_returns_structured_prediction_error(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = EmptyGraphError("0 nodes")
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "prediction_request"})
        error = websocket.receive_json()
        assert error["error"]["code"] == ERROR_PREDICTION_FAILED


def test_unexpected_exception_returns_structured_error(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = RuntimeError("boom")
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "prediction_request"})
        error = websocket.receive_json()
        assert error["error"]["code"] == ERROR_PREDICTION_FAILED
        assert "boom" not in error["error"]["message"]
        assert "Traceback" not in error["error"]["message"]


# --------------------------------------------------------------------------- #
# 4. Client isolation & ping/pong regression
# --------------------------------------------------------------------------- #


def test_multiple_clients_remain_isolated(
    api_client: Any, mock_service: MagicMock
) -> None:
    """A failing request for one client never affects another client."""

    def failing_once(*args: Any, **kwargs: Any) -> PredictionResponse:
        # First call fails; every later call succeeds.
        if mock_service.get_predictions.call_count == 1:
            raise GNNPredictionRuntimeError("boom")
        return make_response(("n0", 1.0))

    mock_service.get_predictions.side_effect = failing_once

    with api_client.websocket_connect(WS_URL) as ws1:
        ws1.receive_json()  # consume connected
        with api_client.websocket_connect(WS_URL) as ws2:
            ws2.receive_json()  # consume connected

            # Client 1 sees its own failure...
            ws1.send_json({"type": "prediction_request"})
            error = ws1.receive_json()
            assert error["error"]["code"] == ERROR_PREDICTION_FAILED

            # ...while client 2 remains fully functional...
            ws2.send_json({"type": "ping"})
            assert ws2.receive_json()["type"] == "pong"
            ws2.send_json({"type": "prediction_request"})
            result = ws2.receive_json()
            assert result["type"] == MESSAGE_TYPE_PREDICTION_RESULT

            # ...and client 1 is still usable after its failed request.
            ws1.send_json({"type": "ping"})
            assert ws1.receive_json()["type"] == "pong"


def test_ping_pong_still_works_around_predictions(
    api_client: Any, mock_service: MagicMock
) -> None:
    """Module 15 ping/pong remains intact before and after predictions."""
    mock_service.get_predictions.return_value = make_response(("n0", 1.0))
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"
        websocket.send_json({"type": "prediction_request"})
        assert websocket.receive_json()["type"] == MESSAGE_TYPE_PREDICTION_RESULT
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_rest_prediction_route_is_still_registered() -> None:
    """Guard: Module 16 does not alter the Module 14 REST API registration."""
    openapi = app.openapi()
    assert "/api/v1/predictions" in openapi["paths"]
    assert "post" in openapi["paths"]["/api/v1/predictions"]