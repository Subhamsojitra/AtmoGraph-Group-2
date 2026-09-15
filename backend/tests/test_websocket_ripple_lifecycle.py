"""Tests for Module 17 Part 6: complete request lifecycle validation.

These tests mount the real ``/api/v1/ws`` route and wire a REAL
:class:`RipplePredictionService` into the WebSocket dependency. Only the
service's leaf dependencies (GraphService, RiskPropagationService,
PredictionService) are mocked, so NO running Neo4j server, trained checkpoint
or GPU is required.

Unlike the Part 4/5 tests (which mock the entire ripple service at the
WebSocket dependency) and the Part 3 tests (which bypass the WebSocket), these
tests exercise the COMPLETE request lifecycle:

    WebSocket request
      -> schema validation
      -> RipplePredictionService.predict
      -> entity resolution (GraphService)
      -> risk propagation (RiskPropagationService)
      -> GNN prediction (PredictionService)
      -> streaming events
      -> final result
      -> completion

Both the success path and the failure paths (entity-not-found, Neo4j
unavailable, GNN failure) are covered.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

# Fake Neo4j env vars must be set before importing the app (pydantic-settings).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

from app.main import app  # noqa: E402
from app.ml.exceptions import GNNPredictionError  # noqa: E402
from app.schemas.prediction import (  # noqa: E402
    NodePrediction,
    PredictionResponse,
)
from app.schemas.risk_propagation import (  # noqa: E402
    AffectedEntity,
    RiskPropagationResponse,
)
from app.schemas.websocket import (  # noqa: E402
    ERROR_INVALID_MESSAGE,
    ERROR_PREDICTION_FAILED,
    MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED,
    MESSAGE_TYPE_RIPPLE_PREDICTION_PROGRESS,
    MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT,
    MESSAGE_TYPE_RIPPLE_PREDICTION_STARTED,
    MESSAGE_TYPE_RIPPLE_DETECTED,
)
from app.services.graph_service import GraphService  # noqa: E402
from app.services.prediction_service import PredictionService  # noqa: E402
from app.services.ripple_prediction.ripple_prediction_service import (  # noqa: E402
    RipplePredictionService,
)
from app.services.risk_propagation.risk_propagation_service import (  # noqa: E402
    RiskPropagationService,
)
from app.services.websocket_manager import reset_connection_manager  # noqa: E402

client = TestClient(app)
WS_URL = "/api/v1/ws"

#: Terminal events for a ripple prediction stream.
_STREAM_TERMINAL_TYPES = frozenset(
    {
        MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED,
        "error",
    }
)


@pytest.fixture(autouse=True)
def _clean_connection_manager() -> None:
    """Start every test with a fresh connection manager registry."""
    reset_connection_manager()
    yield
    reset_connection_manager()


@pytest.fixture()
def mock_graph_service() -> MagicMock:
    """Mock GraphService for entity resolution."""
    return MagicMock(spec=GraphService)


@pytest.fixture()
def mock_risk_propagation_service() -> MagicMock:
    """Mock RiskPropagationService for risk propagation."""
    return MagicMock(spec=RiskPropagationService)


@pytest.fixture()
def mock_prediction_service() -> MagicMock:
    """Mock PredictionService for GNN inference."""
    return MagicMock(spec=PredictionService)


@pytest.fixture()
def real_service(
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
) -> RipplePredictionService:
    """A REAL RipplePredictionService with only its leaf dependencies mocked."""
    return RipplePredictionService(
        graph_service=mock_graph_service,
        risk_propagation_service=mock_risk_propagation_service,
        prediction_service=mock_prediction_service,
    )


@pytest.fixture()
def api_client(real_service: RipplePredictionService) -> Any:
    """A WebSocket client with the REAL ripple service wired in."""
    from app.api.websocket import get_ripple_prediction_service

    app.dependency_overrides[get_ripple_prediction_service] = (
        lambda: real_service
    )
    yield client
    app.dependency_overrides.pop(get_ripple_prediction_service, None)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_affected_entity(
    entity_id: str,
    name: Optional[str] = None,
    depth: int = 1,
    risk_score: float = 50.0,
    risk_level: str = "MEDIUM",
) -> AffectedEntity:
    """Build a Module 10 AffectedEntity."""
    return AffectedEntity(
        entity_id=entity_id,
        entity_name=name or f"Entity {entity_id}",
        depth=depth,
        propagated_risk_score=risk_score,
        propagated_risk_level=risk_level,
    )


def make_propagation_response(
    affected_entities: Optional[list[AffectedEntity]] = None,
) -> RiskPropagationResponse:
    """Build a Module 10 RiskPropagationResponse."""
    if affected_entities is None:
        affected_entities = [
            make_affected_entity("DST_001", "Downstream A", depth=1, risk_score=50.0),
            make_affected_entity("DST_002", "Downstream B", depth=2, risk_score=25.0),
        ]
    return RiskPropagationResponse(
        source_entity_id="SRC_001",
        source_entity_name="Source Entity",
        source_risk_score=75.0,
        affected_entities=affected_entities,
        affected_count=len(affected_entities),
        max_depth_reached=max((e.depth for e in affected_entities), default=0),
        propagated=True,
        error=None,
    )


def make_prediction_response(*pairs: tuple[str, float]) -> PredictionResponse:
    """Build a Module 14 PredictionResponse from (node_id, value) pairs."""
    predictions = [
        NodePrediction(node_id=node_id, prediction=value)
        for node_id, value in pairs
    ]
    return PredictionResponse(
        predictions=predictions, prediction_count=len(predictions)
    )


def mock_node() -> dict[str, Any]:
    """A mock graph node response for entity resolution."""
    return {
        "id": "SRC_001",
        "properties": {"name": "Source Entity", "risk_score": 60.0},
    }


def collect_ripple_stream(websocket: Any) -> list[dict[str, Any]]:
    """Collect all streaming messages for one ripple_prediction request."""
    messages: list[dict[str, Any]] = []
    while True:
        msg = websocket.receive_json()
        messages.append(msg)
        if msg.get("type") in _STREAM_TERMINAL_TYPES:
            break
    return messages


def send_ripple_and_collect(
    websocket: Any, entity_id: str = "SRC_001"
) -> list[dict[str, Any]]:
    """Send a ripple_prediction request and collect the full stream."""
    websocket.send_json(
        {"type": "ripple_prediction", "data": {"entity_id": entity_id}}
    )
    return collect_ripple_stream(websocket)
# --------------------------------------------------------------------------- #
# 1. Complete success lifecycle
# --------------------------------------------------------------------------- #


def test_complete_lifecycle_streams_real_result(
    api_client: Any,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
) -> None:
    """The full request lifecycle streams real data end to end.

    The REAL RipplePredictionService runs entity resolution, risk propagation
    and GNN prediction (with mocked leaves) and the WebSocket layer streams
    the full lifecycle: started, progress, detected per real entity, result,
    completed.
    """
    mock_graph_service.get_node_by_id.return_value = mock_node()
    mock_risk_propagation_service.propagate.return_value = (
        make_propagation_response()
    )
    mock_prediction_service.get_predictions.return_value = (
        make_prediction_response(("DST_001", 0.75), ("DST_002", 0.30))
    )

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    types = [m["type"] for m in messages]
    # Lifecycle order: started -> progress -> detected -> progress -> result -> completed
    assert types[0] == MESSAGE_TYPE_RIPPLE_PREDICTION_STARTED
    assert types[-1] == MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED
    assert MESSAGE_TYPE_RIPPLE_PREDICTION_PROGRESS in types
    result_idx = types.index(MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT)
    assert result_idx < len(types) - 1

    # The service was driven through the real pipeline.
    assert mock_graph_service.get_node_by_id.call_count == 1
    assert mock_risk_propagation_service.propagate.call_count == 1
    assert mock_prediction_service.get_predictions.call_count == 1

    # ripple_detected only for the REAL affected entities.
    detected = [
        m for m in messages if m["type"] == MESSAGE_TYPE_RIPPLE_DETECTED
    ]
    assert len(detected) == 2
    assert detected[0]["data"]["entity_id"] == "DST_001"
    assert detected[1]["data"]["entity_id"] == "DST_002"

    # The result carries the real service output (real GNN predictions).
    result_msg = next(
        m
        for m in messages
        if m["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT
    )
    data = result_msg["data"]
    assert data["source_entity_id"] == "SRC_001"
    assert data["affected_count"] == 2
    assert data["prediction_count"] == 2
    pred_map = {
        e["entity_id"]: e["gnn_prediction"]
        for e in data["affected_entities"]
    }
    assert pred_map["DST_001"] == 0.75
    assert pred_map["DST_002"] == 0.30

    # Completed event reflects the real counts.
    completed = messages[-1]
    assert completed["data"]["source_entity_id"] == "SRC_001"
    assert completed["data"]["affected_count"] == 2
    assert completed["data"]["prediction_count"] == 2


def test_complete_lifecycle_result_matches_real_service(
    api_client: Any,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
) -> None:
    """The streamed result equals the REAL service result exactly.

    Only the entities returned by the (mocked) propagation engine appear, and
    only the predictions returned by the (mocked) GNN service are attached.
    """
    mock_graph_service.get_node_by_id.return_value = mock_node()
    mock_risk_propagation_service.propagate.return_value = (
        make_propagation_response(
            affected_entities=[
                make_affected_entity("REAL_A", depth=1, risk_score=60.0),
                make_affected_entity("REAL_B", depth=2, risk_score=20.0),
            ],
        )
    )
    mock_prediction_service.get_predictions.return_value = (
        make_prediction_response(
            ("REAL_A", 0.9),
            ("REAL_B", 0.1),
            ("GHOST", 0.99),  # not in the propagation output
        )
    )

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    result_msg = next(
        m
        for m in messages
        if m["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT
    )
    data = result_msg["data"]
    entity_ids = [e["entity_id"] for e in data["affected_entities"]]

    # GHOST must never be fabricated into the result.
    assert entity_ids == ["REAL_A", "REAL_B"]
    pred_map = {
        e["entity_id"]: e["gnn_prediction"]
        for e in data["affected_entities"]
    }
    assert pred_map["REAL_A"] == 0.9
    assert pred_map["REAL_B"] == 0.1


# --------------------------------------------------------------------------- #
# 2. Complete failure paths
# --------------------------------------------------------------------------- #


def test_lifecycle_entity_not_found_returns_structured_error(
    api_client: Any,
    mock_graph_service: MagicMock,
) -> None:
    """Entity resolution returning no node fails with a structured error.

    The REAL service raises EntityNotFoundError, which the WebSocket layer
    maps to a client-safe INVALID_MESSAGE error. No false completion follows.
    """
    mock_graph_service.get_node_by_id.return_value = None

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    # The stream starts with started, then carries the structured error.
    assert messages[0]["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_STARTED
    assert messages[-1]["type"] == "error"
    assert messages[-1]["error"]["code"] == ERROR_INVALID_MESSAGE
    # No completion event after a failure.
    assert not any(
        m["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED for m in messages
    )


def test_lifecycle_neo4j_unavailable_returns_structured_error(
    api_client: Any,
    mock_graph_service: MagicMock,
) -> None:
    """Neo4j failing during entity resolution yields a structured error."""
    mock_graph_service.get_node_by_id.side_effect = ServiceUnavailable(
        "neo4j://test-host:7687"
    )

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    error = messages[-1]
    assert error["type"] == "error"
    assert error["error"]["code"] == ERROR_PREDICTION_FAILED
    # Client-safe: no internal URI or internals exposed.
    assert "neo4j://test-host:7687" not in error["error"]["message"]
    assert not any(
        m["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED for m in messages
    )


def test_lifecycle_gnn_failure_returns_structured_error(
    api_client: Any,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
) -> None:
    """GNN inference failure surfaces as a structured PREDICTION_FAILED error."""
    mock_graph_service.get_node_by_id.return_value = mock_node()
    mock_risk_propagation_service.propagate.return_value = (
        make_propagation_response()
    )
    mock_prediction_service.get_predictions.side_effect = GNNPredictionError(
        "inference crashed"
    )

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    error = messages[-1]
    assert error["type"] == "error"
    assert error["error"]["code"] == ERROR_PREDICTION_FAILED
    assert "inference crashed" not in error["error"]["message"]
    assert not any(
        m["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED for m in messages
    )


def test_lifecycle_no_checkpoint_degrades_gracefully(
    api_client: Any,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
) -> None:
    """No GNN checkpoint yields a result WITHOUT predictions (never an error).

    The existing RipplePredictionService treats a missing checkpoint as a
    non-fatal condition: it returns the real propagation data with
    ``gnn_prediction`` left None rather than failing the whole request.
    """
    mock_graph_service.get_node_by_id.return_value = mock_node()
    mock_risk_propagation_service.propagate.return_value = (
        make_propagation_response()
    )
    mock_prediction_service.get_predictions.return_value = (
        make_prediction_response()
    )

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    assert messages[-1]["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED
    result_msg = next(
        m
        for m in messages
        if m["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT
    )
    data = result_msg["data"]
    assert data["affected_count"] == 2
    assert data["prediction_count"] == 0
    assert all(
        e["gnn_prediction"] is None for e in data["affected_entities"]
    )


def test_lifecycle_zero_affected_entities_no_detected_events(
    api_client: Any,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
) -> None:
    """Zero affected entities produces ZERO ripple_detected events."""
    mock_graph_service.get_node_by_id.return_value = mock_node()
    mock_risk_propagation_service.propagate.return_value = (
        make_propagation_response(affected_entities=[])
    )
    mock_prediction_service.get_predictions.return_value = (
        make_prediction_response()
    )

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    detected = [
        m for m in messages if m["type"] == MESSAGE_TYPE_RIPPLE_DETECTED
    ]
    assert len(detected) == 0
    assert messages[-1]["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED
    assert messages[-1]["data"]["affected_count"] == 0


def test_lifecycle_failure_keeps_connection_alive(
    api_client: Any,
    mock_graph_service: MagicMock,
) -> None:
    """After a failed lifecycle the connection still handles ping/pong."""
    mock_graph_service.get_node_by_id.return_value = None

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)
        assert messages[-1]["type"] == "error"

        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_lifecycle_multiple_clients_remain_isolated(
    api_client: Any,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
) -> None:
    """One failing client does not corrupt another client's lifecycle."""
    def get_node_by_id(entity_id: str) -> Any:
        if entity_id == "MISSING":
            return None
        return mock_node()

    mock_graph_service.get_node_by_id.side_effect = get_node_by_id
    mock_risk_propagation_service.propagate.return_value = (
        make_propagation_response()
    )
    mock_prediction_service.get_predictions.return_value = (
        make_prediction_response(("DST_001", 0.75), ("DST_002", 0.30))
    )

    with api_client.websocket_connect(WS_URL) as ws1:
        ws1.receive_json()  # consume connected
        with api_client.websocket_connect(WS_URL) as ws2:
            ws2.receive_json()  # consume connected

            # Client 1 requests a missing entity -> structured error.
            ws1_messages = send_ripple_and_collect(ws1, entity_id="MISSING")
            assert ws1_messages[-1]["type"] == "error"
            assert ws1_messages[-1]["error"]["code"] == ERROR_INVALID_MESSAGE

            # Client 2 still completes a full, valid lifecycle.
            ws2_messages = send_ripple_and_collect(ws2)
            assert (
                ws2_messages[-1]["type"]
                == MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED
            )
            assert any(
                m["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT
                for m in ws2_messages
            )

            # Client 1 remains usable after its failure.
            ws1.send_json({"type": "ping"})
            assert ws1.receive_json()["type"] == "pong"