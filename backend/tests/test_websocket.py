"""Tests for the Module 15 WebSocket foundation.

These tests exercise the raw transport endpoint with FastAPI's TestClient
WebSocket support. They deliberately require NO running Neo4j server and NO
GNN model: establishing a connection, pinging and receiving pong never touch
the database or the ML stack.

Conventions follow ``tests/test_health.py`` and the other API test modules:
fake Neo4j environment variables are set before importing the app so the
central configuration can load (the WebSocket transport itself never reads
them).
"""

from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.routing import WebSocketRoute

# Fake Neo4j env vars must be set before importing the app (pydantic-settings).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

from app.api.websocket import (  # noqa: E402
    WEBSOCKET_ROUTE_PATH,
    create_response_for_message,
)
from app.main import app  # noqa: E402
from app.schemas.websocket import (  # noqa: E402
    ERROR_INVALID_JSON,
    ERROR_INVALID_MESSAGE,
    ERROR_NOT_SUPPORTED_YET,
    ERROR_UNSUPPORTED_MESSAGE_TYPE,
    InboundWebSocketMessage,
    build_error_message,
)
from app.services.websocket_manager import (  # noqa: E402
    ConnectionManager,
    reset_connection_manager,
)

client = TestClient(app)

WS_URL = "/api/v1/ws"

# Absolute paths of the Module 15 transport modules (used by the no-dependency
# design tests).
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_TRANSPORT_MODULE_PATHS = [
    _BACKEND_DIR / "app" / "api" / "websocket.py",
    _BACKEND_DIR / "app" / "schemas" / "websocket.py",
    _BACKEND_DIR / "app" / "services" / "websocket_manager.py",
]


@pytest.fixture(autouse=True)
def _clean_connection_manager() -> None:
    """Start every test with a fresh connection manager registry."""
    reset_connection_manager()
    yield
    reset_connection_manager()


# --------------------------------------------------------------------------- #
# 1. Routing / registration
# --------------------------------------------------------------------------- #


def test_websocket_route_is_registered_in_app() -> None:
    """The WebSocket route is present in the runtime routing table."""
    routes = {route.path: type(route) for route in app.routes}
    assert WS_URL in routes
    # FastAPI serves @router.websocket routes through its own
    # APIWebSocketRoute, a subclass of Starlette's WebSocketRoute.
    assert issubclass(routes[WS_URL], WebSocketRoute)


def test_websocket_route_is_not_in_openapi() -> None:
    """FastAPI does not expose WebSocket routes through OpenAPI/ Swagger.

    Absence from ``/openapi.json`` is expected and is NOT a sign of a broken
    route; the routing-table test above is the authoritative check.
    """
    openapi = app.openapi()
    assert WEBSOCKET_ROUTE_PATH not in openapi["paths"]
    assert WS_URL not in openapi["paths"]


# --------------------------------------------------------------------------- #
# 2. Connection lifecycle
# --------------------------------------------------------------------------- #


def test_connection_succeeds_and_sends_connected_message() -> None:
    """A client connects and immediately receives the ``connected`` message."""
    with client.websocket_connect(WS_URL) as websocket:
        message = websocket.receive_json()
        assert message["type"] == "connected"
        assert "client_id" in message["data"]
        assert message["data"]["protocol"] >= 1
        assert "ping" in message["data"]["supported_client_messages"]


def test_client_disconnect_is_handled_and_server_stays_healthy() -> None:
    """A clean disconnect does not crash the server; reconnects still work."""
    with client.websocket_connect(WS_URL) as websocket:
        connected = websocket.receive_json()
        assert connected["type"] == "connected"

    # The server is still healthy: a brand-new connection succeeds.
    with client.websocket_connect(WS_URL) as websocket:
        assert websocket.receive_json()["type"] == "connected"


def test_multiple_clients_can_connect_simultaneously() -> None:
    """Several clients can hold connections at the same time."""
    with client.websocket_connect(WS_URL) as ws1:
        with client.websocket_connect(WS_URL) as ws2:
            assert ws1.receive_json()["type"] == "connected"
            assert ws2.receive_json()["type"] == "connected"


# --------------------------------------------------------------------------- #
# 3. Message round-trips
# --------------------------------------------------------------------------- #


def test_ping_gets_pong() -> None:
    """A valid ``ping`` message receives a ``pong`` response."""
    with client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "ping"})
        response = websocket.receive_json()
        assert response["type"] == "pong"
        # Pong envelopes carry no error payload; the key is omitted entirely.
        assert response.get("error") is None


def test_ping_with_data_is_echoed() -> None:
    """A ping carrying a data object echoes it back in the pong."""
    with client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "ping", "data": {"trace": "abc-123"}})
        response = websocket.receive_json()
        assert response["type"] == "pong"
        assert response["data"]["echo"] == {"trace": "abc-123"}


# --------------------------------------------------------------------------- #
# 4. Invalid input handling (connection must stay usable)
# --------------------------------------------------------------------------- #


def test_malformed_json_returns_structured_error_and_connection_survives() -> None:
    """Unparseable JSON yields INVALID_JSON and does not kill the connection."""
    with client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_text("{not valid json")
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_INVALID_JSON
        assert error["error"]["message"]

        # The connection is still usable.
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_non_object_json_returns_structured_error() -> None:
    """A JSON array at the top level is rejected as INVALID_MESSAGE."""
    with client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_text("[1, 2, 3]")
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_INVALID_MESSAGE


def test_unknown_message_type_returns_structured_error() -> None:
    """An unsupported message type is rejected with a clear error."""
    with client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "make_coffee"})
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_UNSUPPORTED_MESSAGE_TYPE
        assert "make_coffee" in error["error"]["message"]

        # The connection is still usable.
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_missing_type_returns_structured_error() -> None:
    """A message without a ``type`` field is rejected by Pydantic."""
    with client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"data": {"x": 1}})
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_INVALID_MESSAGE


def test_extra_fields_are_rejected() -> None:
    """Unknown top-level fields are rejected (strict transport contract)."""
    with client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "ping", "surprise": "field"})
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_INVALID_MESSAGE


@pytest.mark.parametrize(
    "message_type", ["prediction_request", "ripple_prediction"]
)
def test_reserved_module_16_17_types_are_not_handled_yet(
    message_type: str,
) -> None:
    """Module 15 recognizes future messages but does not run the ML pipeline."""
    with client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": message_type})
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_NOT_SUPPORTED_YET


def test_disconnect_of_one_client_does_not_break_another() -> None:
    """Closing one socket leaves the remaining clients fully functional."""
    with client.websocket_connect(WS_URL) as ws2:
        assert ws2.receive_json()["type"] == "connected"

        with client.websocket_connect(WS_URL) as ws1:
            assert ws1.receive_json()["type"] == "connected"

        # ws1 has disconnected here; ws2 must remain fully functional.
        ws2.send_json({"type": "ping"})
        assert ws2.receive_json()["type"] == "pong"


# --------------------------------------------------------------------------- #
# 5. Pure message-processing helpers (no socket)
# --------------------------------------------------------------------------- #


def test_message_processor_rejects_bad_json() -> None:
    response = create_response_for_message("{oops")
    assert response["type"] == "error"
    assert response["error"]["code"] == ERROR_INVALID_JSON
    assert "stack" not in response["error"]["message"].lower()


def test_message_processor_rejects_non_object_payload() -> None:
    response = create_response_for_message("42")
    assert response["type"] == "error"
    assert response["error"]["code"] == ERROR_INVALID_MESSAGE


def test_message_processor_ping_returns_pong() -> None:
    response = create_response_for_message('{"type": "ping"}')
    assert response["type"] == "pong"


def test_inbound_schema_rejects_blank_type() -> None:
    with pytest.raises(ValidationError):
        InboundWebSocketMessage(type="   ")


def test_inbound_schema_rejects_non_object_data() -> None:
    with pytest.raises(ValidationError):
        InboundWebSocketMessage(type="ping", data=[1, 2])


def test_error_envelopes_are_strict() -> None:
    """Errors carry only the documented fields -- no internals."""
    envelope = build_error_message("X", "boom")
    assert set(envelope.keys()) == {"type", "timestamp", "error"}
    assert envelope["error"]["code"] == "X"


# --------------------------------------------------------------------------- #
# 6. Connection manager behaviour
# --------------------------------------------------------------------------- #


class _FakeWebSocket:
    """Minimal stand-in for a FastAPI WebSocket used by manager tests."""

    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True


class _DeadWebSocket(_FakeWebSocket):
    """A fake socket whose peer has already gone away."""

    async def send_json(self, payload: dict[str, Any]) -> None:
        raise RuntimeError("connection is closed")


def test_connection_manager_register_send_broadcast_remove() -> None:
    """The manager registers, sends, broadcasts and removes connections."""
    manager = ConnectionManager()
    websocket = _FakeWebSocket()

    asyncio.run(manager.connect(websocket, "client-a"))
    assert websocket.accepted is True
    assert manager.active_connections == 1
    assert manager.is_connected("client-a")

    delivered = asyncio.run(manager.broadcast_json({"type": "ping"}))
    assert delivered == 1
    assert websocket.sent == [{"type": "ping"}]

    asyncio.run(manager.disconnect("client-a"))
    assert manager.active_connections == 0
    assert not manager.is_connected("client-a")


def test_connection_manager_send_to_unknown_client_returns_false() -> None:
    """Sending to an unknown client id reports failure instead of raising."""
    manager = ConnectionManager()
    ok = asyncio.run(manager.send_json("ghost", {"type": "ping"}))
    assert ok is False


def test_connection_manager_removes_dead_connection_on_send_failure() -> None:
    """A failed send prunes the stale connection from the registry."""
    manager = ConnectionManager()
    dead = _DeadWebSocket()
    asyncio.run(manager.connect(dead, "dead-client"))
    assert manager.active_connections == 1

    ok = asyncio.run(manager.send_json("dead-client", {"type": "ping"}))
    assert ok is False
    assert manager.active_connections == 0


def test_connection_manager_replace_same_client_id() -> None:
    """Registering the same id twice replaces the previous socket."""
    manager = ConnectionManager()
    first = _FakeWebSocket()
    second = _FakeWebSocket()
    asyncio.run(manager.connect(first, "client-a"))
    asyncio.run(manager.connect(second, "client-a"))
    assert manager.active_connections == 1
    asyncio.run(manager.broadcast_json({"type": "ping"}))
    assert first.sent == []
    assert second.sent == [{"type": "ping"}]


def test_connection_manager_close_all() -> None:
    """close_all clears the registry and closes every socket."""
    manager = ConnectionManager()
    ws_a = _FakeWebSocket()
    ws_b = _FakeWebSocket()
    asyncio.run(manager.connect(ws_a, "client-a"))
    asyncio.run(manager.connect(ws_b, "client-b"))

    asyncio.run(manager.close_all())
    assert manager.active_connections == 0
    assert ws_a.closed is True
    assert ws_b.closed is True


# --------------------------------------------------------------------------- #
# 7. No database / no ML dependency
# --------------------------------------------------------------------------- #


def test_connection_and_ping_require_no_neo4j() -> None:
    """Connecting and pinging works with NO database running at all."""
    with client.websocket_connect(WS_URL) as websocket:
        assert websocket.receive_json()["type"] == "connected"
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_connection_requires_no_gnn_model_to_be_loaded() -> None:
    """Connecting does not load or require the GNN model / checkpoint."""
    with client.websocket_connect(WS_URL) as websocket:
        assert websocket.receive_json()["type"] == "connected"
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_transport_modules_do_not_import_neo4j_or_ml() -> None:
    """The Module 15 transport layer has no graph/ML imports.

    This enforces the design rule that the WebSocket foundation must not pull
    Neo4j or the GNN stack into the connection path. Import statements are
    parsed with :mod:`ast` so prose mentions of these systems in docstrings
    do not cause false positives.
    """
    banned_roots = {"neo4j", "torch"}
    banned_app_prefixes = (
        "app.ml",
        "app.database",
        "app.repositories",
        "app.services.graph",
        "app.services.risk",
        "app.services.prediction",
        "app.services.nlp",
        "app.services.integration",
    )
    for path in _TRANSPORT_MODULE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in banned_roots, (
                        f"{path.name} must not import {alias.name!r}"
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                assert root not in banned_roots, (
                    f"{path.name} must not import from {module!r}"
                )
                if root == "app":
                    assert not module.startswith(banned_app_prefixes), (
                        f"{path.name} must not import from {module!r}"
                    )