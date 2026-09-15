"""Write the Part 4 WebSocket ripple prediction test file (part 3)."""

content = '''

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
'''

with open(
    'D:/Infotact_projects/AtmoGraph-Group-2/backend/tests/'
    'test_websocket_ripple_prediction.py',
    'a',
) as f:
    f.write(content)

print('Part 3 written')
