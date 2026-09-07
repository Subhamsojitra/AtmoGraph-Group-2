"""Write the Part 4 WebSocket ripple prediction test file (part 2)."""

content = '''

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
'''

with open(
    'D:/Infotact_projects/AtmoGraph-Group-2/backend/tests/'
    'test_websocket_ripple_prediction.py',
    'a',
) as f:
    f.write(content)

print('Part 2 written')
