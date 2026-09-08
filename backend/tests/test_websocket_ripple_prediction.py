"""Tests for Module 17 Part 4 & 5: WebSocket -> RipplePredictionService integration
and real-time streaming events.

These tests mount the real ``/api/v1/ws`` route but MOCK the ripple prediction
service at the WebSocket dependency, so NONE of them require a running Neo4j
server, a trained GNN checkpoint or a GPU. They verify the Module 17 request
dispatch, streaming lifecycle, response normalization, error mapping and
client isolation around the EXISTING Module 17 service contract.

The streaming lifecycle for a valid ripple_prediction request is:

    ripple_prediction_started
    ripple_prediction_progress  (risk_propagation stage)
    ripple_detected             (one per REAL affected entity)
    ripple_prediction_progress  (gnn_prediction stage)
    ripple_prediction_result
    ripple_prediction_completed

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
    ERROR_MODEL_UNAVAILABLE,
    ERROR_PREDICTION_FAILED,
    MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED,
    MESSAGE_TYPE_RIPPLE_PREDICTION_PROGRESS,
    MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT,
    MESSAGE_TYPE_RIPPLE_PREDICTION_STARTED,
    MESSAGE_TYPE_RIPPLE_DETECTED,
)
from app.services.prediction_service import ModelNotAvailableError  # noqa: E402
from app.services.ripple_prediction.exceptions import RipplePredictionError
from app.services.ripple_prediction.result_schema import (
    RippleAffectedEntity,
    RipplePredictionResult,
)
from app.services.risk.exceptions import EntityNotFoundError
from app.services.websocket_manager import reset_connection_manager  # noqa: E402
from neo4j.exceptions import ServiceUnavailable  # noqa: E402

client = TestClient(app)
WS_URL = "/api/v1/ws"

#: Message types that terminate a ripple prediction stream.
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
    if affected_entities is None:
        affected_entities = [
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
        affected_entities=affected_entities,
        affected_count=len(affected_entities),
        max_depth_reached=max((e.depth for e in affected_entities), default=0),
        prediction_count=sum(
            1 for e in affected_entities if e.gnn_prediction is not None
        ),
        error=error,
    )


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


def test_valid_ripple_prediction_streams_full_lifecycle(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """A valid ripple_prediction streams the complete event lifecycle."""
    mock_ripple_service.predict.return_value = make_ripple_result()

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    types = [m["type"] for m in messages]
    assert types[0] == MESSAGE_TYPE_RIPPLE_PREDICTION_STARTED
    assert types[-1] == MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED
    assert MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT in types
    assert MESSAGE_TYPE_RIPPLE_PREDICTION_PROGRESS in types


def test_ripple_prediction_started_is_emitted_first(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """ripple_prediction_started is the first event in the stream."""
    mock_ripple_service.predict.return_value = make_ripple_result()

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    started = messages[0]
    assert started["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_STARTED
    assert started["data"]["entity_id"] == "SRC_001"


def test_ripple_prediction_result_contains_real_service_result(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """ripple_prediction_result carries the real RipplePredictionResult."""
    mock_ripple_service.predict.return_value = make_ripple_result()

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    result_msg = next(
        m for m in messages if m["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT
    )
    data = result_msg["data"]
    assert data["source_entity_id"] == "SRC_001"
    assert data["affected_count"] == 1
    assert data["prediction_count"] == 1
    assert len(data["affected_entities"]) == 1
    assert data["affected_entities"][0]["entity_id"] == "DST_001"


def test_ripple_prediction_completed_is_emitted_last(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """ripple_prediction_completed is the final event on success."""
    mock_ripple_service.predict.return_value = make_ripple_result()

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    completed = messages[-1]
    assert completed["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED
    assert completed["data"]["source_entity_id"] == "SRC_001"
    assert completed["data"]["affected_count"] == 1
    assert completed["data"]["prediction_count"] == 1


def test_ripple_detected_emitted_for_real_affected_entities(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """A ripple_detected event is emitted for EACH real affected entity."""
    mock_ripple_service.predict.return_value = make_ripple_result(
        affected_entities=[
            RippleAffectedEntity(
                entity_id="DST_001",
                entity_name="Downstream A",
                depth=1,
                propagated_risk_score=50.0,
                propagated_risk_level="MEDIUM",
                gnn_prediction=0.75,
            ),
            RippleAffectedEntity(
                entity_id="DST_002",
                entity_name="Downstream B",
                depth=2,
                propagated_risk_score=30.0,
                propagated_risk_level="LOW",
                gnn_prediction=0.60,
            ),
        ],
    )

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    detected = [
        m for m in messages if m["type"] == MESSAGE_TYPE_RIPPLE_DETECTED
    ]
    assert len(detected) == 2
    assert detected[0]["data"]["entity_id"] == "DST_001"
    assert detected[1]["data"]["entity_id"] == "DST_002"


def test_zero_affected_entities_generates_no_ripple_detected(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """Zero affected entities means ZERO ripple_detected events."""
    mock_ripple_service.predict.return_value = make_ripple_result(
        affected_entities=[],
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


def test_ripple_prediction_service_receives_validated_request(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """The service receives the validated WebSocketRipplePredictionRequest."""
    mock_ripple_service.predict.return_value = make_ripple_result()

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        send_ripple_and_collect(websocket)

    assert mock_ripple_service.predict.call_count == 1
    request = mock_ripple_service.predict.call_args[0][0]
def test_ripple_prediction_missing_entity_id_returns_invalid_message(
    api_client: Any,
) -> None:
    """A ripple_prediction with no entity_id fails validation."""
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "ripple_prediction", "data": {}})
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["error"]["code"] == ERROR_INVALID_MESSAGE


def test_ripple_prediction_blank_entity_id_returns_invalid_message(
    api_client: Any,
) -> None:
    """A blank/whitespace entity_id fails validation."""
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "ripple_prediction", "data": {"entity_id": "   "}}
        )
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["error"]["code"] == ERROR_INVALID_MESSAGE


def test_ripple_prediction_entity_not_found_returns_structured_error(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """EntityNotFoundError maps to a structured INVALID_MESSAGE error."""
    mock_ripple_service.predict.side_effect = EntityNotFoundError("nope")

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    assert messages[0]["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_STARTED
    error = next(m for m in messages if m["type"] == "error")
    assert error["error"]["code"] == ERROR_INVALID_MESSAGE
    assert messages[-1]["type"] == "error"


def test_ripple_prediction_model_unavailable_returns_structured_error(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """ModelNotAvailableError maps to a structured MODEL_UNAVAILABLE error."""
    mock_ripple_service.predict.side_effect = ModelNotAvailableError(
        "no checkpoint"
    )

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    error = next(m for m in messages if m["type"] == "error")
    assert error["error"]["code"] == ERROR_MODEL_UNAVAILABLE
    # No false completion after an error.
    assert messages[-1]["type"] == "error"


def test_ripple_prediction_neo4j_unavailable_returns_structured_error(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """Neo4j ServiceUnavailable maps to a structured PREDICTION_FAILED error."""
    mock_ripple_service.predict.side_effect = ServiceUnavailable(
        "neo4j://test-host:7687"
    )

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    error = next(m for m in messages if m["type"] == "error")
    assert error["error"]["code"] == ERROR_PREDICTION_FAILED
    # The message is client-safe (no credentials / internals).
    assert "neo4j://test-host:7687" not in error["error"]["message"]
    assert messages[-1]["type"] == "error"


def test_ripple_prediction_service_failure_returns_structured_error(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """RipplePredictionError maps to a structured PREDICTION_FAILED error."""
    mock_ripple_service.predict.side_effect = RipplePredictionError("boom")

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    error = next(m for m in messages if m["type"] == "error")
    assert error["error"]["code"] == ERROR_PREDICTION_FAILED
    assert "boom" not in error["error"]["message"]


def test_ripple_prediction_unexpected_exception_returns_structured_error(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """An unexpected exception maps to a structured error, no traceback."""
    mock_ripple_service.predict.side_effect = RuntimeError("boom")

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    error = next(m for m in messages if m["type"] == "error")
    assert error["error"]["code"] == ERROR_PREDICTION_FAILED
    assert "boom" not in error["error"]["message"]
    assert "Traceback" not in error["error"]["message"]


def test_ripple_prediction_failure_keeps_connection_alive(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """A failing ripple_prediction does NOT terminate the connection."""
    mock_ripple_service.predict.side_effect = RipplePredictionError("boom")

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

        assert messages[-1]["type"] == "error"

        # Connection is still alive: ping/pong works
        websocket.send_json({"type": "ping"})
def test_ping_pong_still_works_around_ripple_predictions(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """Module 15 ping/pong remains intact before and after ripple predictions."""
    mock_ripple_service.predict.return_value = make_ripple_result()

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"
        messages = send_ripple_and_collect(websocket)
        assert any(
            m["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT
            for m in messages
        )
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
            ws1_messages = send_ripple_and_collect(ws1)
            assert ws1_messages[-1]["type"] == "error"
            assert ws1_messages[-1]["error"]["code"] == ERROR_PREDICTION_FAILED

            # ...while client 2 remains fully functional...
            ws2.send_json({"type": "ping"})
            assert ws2.receive_json()["type"] == "pong"
            ws2_messages = send_ripple_and_collect(ws2)
            assert any(
                m["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT
                for m in ws2_messages
            )

            # ...and client 1 is still usable after its failed request.
            ws1.send_json({"type": "ping"})
            assert ws1.receive_json()["type"] == "pong"


def test_progress_events_use_real_stages_not_fabricated_percentages(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """Progress events use meaningful stage names, not fabricated percentages."""
    mock_ripple_service.predict.return_value = make_ripple_result()

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    progress_events = [
        m for m in messages
        if m["type"] == MESSAGE_TYPE_RIPPLE_PREDICTION_PROGRESS
    ]
    assert len(progress_events) >= 2
    stages = [m["data"]["stage"] for m in progress_events]
    assert "risk_propagation" in stages
    assert "gnn_prediction" in stages
    for pe in progress_events:
        assert "percent" not in pe["data"]
        assert "percentage" not in pe["data"]


def test_stream_ordering_follows_lifecycle(
    api_client: Any, mock_ripple_service: MagicMock
) -> None:
    """Events follow the expected lifecycle ordering."""
    mock_ripple_service.predict.return_value = make_ripple_result(
        affected_entities=[
            RippleAffectedEntity(
                entity_id="DST_001",
                entity_name="A",
                depth=1,
                propagated_risk_score=50.0,
                propagated_risk_level="MEDIUM",
                gnn_prediction=0.75,
            ),
            RippleAffectedEntity(
                entity_id="DST_002",
                entity_name="B",
                depth=2,
                propagated_risk_score=30.0,
                propagated_risk_level="LOW",
                gnn_prediction=0.60,
            ),
        ],
    )

    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        messages = send_ripple_and_collect(websocket)

    types = [m["type"] for m in messages]
    assert types[0] == MESSAGE_TYPE_RIPPLE_PREDICTION_STARTED
    assert types[-1] == MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED
    result_idx = types.index(MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT)
    completed_idx = len(types) - 1
    assert result_idx < completed_idx
    detected_indices = [
        i for i, t in enumerate(types) if t == MESSAGE_TYPE_RIPPLE_DETECTED
    ]
    assert len(detected_indices) == 2
    started_idx = 0
    assert all(i > started_idx for i in detected_indices)
    assert all(i < result_idx for i in detected_indices)
