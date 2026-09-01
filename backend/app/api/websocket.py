"""WebSocket endpoint for AtmoGraph (Module 15).

Establishes the in-process WebSocket foundation that Modules 16/17 will extend
with real-time ML / ripple-effect prediction streaming. This module implements
transport concerns only:

    * connection lifecycle (accept -> connected -> loop -> disconnect)
    * JSON receive / send
    * Pydantic-driven message validation
    * structured, client-safe errors (no stack traces / paths / credentials)

The endpoint deliberately has NO database or ML dependency: establishing a
connection, pinging and receiving pong never touches Neo4j or the GNN model.

Error codes (see :mod:`app.schemas.websocket`):

    INVALID_JSON                 unparseable JSON text
    INVALID_MESSAGE              not an object / schema violation
    UNSUPPORTED_MESSAGE_TYPE     unknown message type
    NOT_SUPPORTED_YET            reserved for Modules 16/17
    INTERNAL_ERROR               unexpected server-side failure
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.logger import get_logger
from app.schemas.websocket import (
    ERROR_INTERNAL,
    ERROR_INVALID_JSON,
    ERROR_INVALID_MESSAGE,
    ERROR_NOT_SUPPORTED_YET,
    ERROR_UNSUPPORTED_MESSAGE_TYPE,
    MESSAGE_TYPE_PING,
    RESERVED_CLIENT_MESSAGE_TYPES,
    SUPPORTED_CLIENT_MESSAGE_TYPES,
    InboundWebSocketMessage,
    build_connected_message,
    build_error_message,
    build_pong_message,
)
from app.services.websocket_manager import (
    ConnectionManager,
    get_connection_manager,
)

logger = get_logger(__name__)

router = APIRouter(tags=["websocket"])

#: Route path (relative to the ``/api/v1`` prefix applied in ``app.main``).
#: The full endpoint is ``/api/v1/ws``.
WEBSOCKET_ROUTE_PATH = "/ws"


# --------------------------------------------------------------------------- #
# Message processing (pure, socket-free helpers - unit-testable)
# --------------------------------------------------------------------------- #


def create_response_for_message(raw_text: str) -> dict[str, Any]:
    """Validate one raw client message and return the envelope to send back.

    This is a pure, network-free function: it never raises, so an endpoint
    loop can pipe any client input into it and always receive a response --
    either the requested ``pong`` or a structured ``error``.

    Args:
        raw_text: The raw text frame received from the client.

    Returns:
        A JSON-serializable response envelope (``pong`` or ``error``).
    """
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return build_error_message(
            ERROR_INVALID_JSON, "Malformed JSON payload."
        )

    if not isinstance(payload, dict):
        return build_error_message(
            ERROR_INVALID_MESSAGE,
            "The message must be a JSON object at the top level.",
        )

    try:
        message = InboundWebSocketMessage(**payload)
    except (ValidationError, TypeError) as exc:
        return build_error_message(
            ERROR_INVALID_MESSAGE, _summarize_validation_error(exc)
        )

    try:
        return dispatch_message(message)
    except Exception:
        logger.error(
            "Unexpected error while dispatching a WebSocket message",
            exc_info=True,
        )
        return build_error_message(
            ERROR_INTERNAL, "An unexpected server error occurred."
        )


def dispatch_message(message: InboundWebSocketMessage) -> dict[str, Any]:
    """Dispatch a validated inbound message to its Module 15 handler.

    Args:
        message: The validated message.

    Returns:
        The appropriate response envelope (``pong`` or ``error``).
    """
    message_type = message.type
    if message_type == MESSAGE_TYPE_PING:
        return build_pong_message(message.data)
    if message_type in RESERVED_CLIENT_MESSAGE_TYPES:
        return build_error_message(
            ERROR_NOT_SUPPORTED_YET,
            f"Message type '{message_type}' is reserved for the ML streaming "
            "pipeline (Modules 16/17) and is not handled yet.",
        )
    supported = ", ".join(sorted(SUPPORTED_CLIENT_MESSAGE_TYPES))
    return build_error_message(
        ERROR_UNSUPPORTED_MESSAGE_TYPE,
        f"Unsupported message type '{message_type}'. "
        f"Supported types: {supported}.",
    )


def _summarize_validation_error(exc: ValidationError) -> str:
    """Turn a Pydantic :class:`ValidationError` into a short client-safe text.

    Args:
        exc: The validation error raised for the inbound message.

    Returns:
        A one-line description of the first failing field.
    """
    errors = exc.errors()
    if not errors:
        return "Invalid message."
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    reason = first.get("msg", "invalid value")
    return f"Invalid field '{location}': {reason}." if location else reason


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #


@router.websocket(WEBSOCKET_ROUTE_PATH)
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Serve the Module 15 WebSocket transport at ``/api/v1/ws``.

    Lifecycle:

        1. accept the connection and register it with the connection manager
        2. send the ``connected`` message (client_id + protocol info)
        3. loop: receive one text frame -> validate -> respond
        4. on disconnect / unexpected failure: remove the client, then close

    One faulty client (bad JSON, unknown type, abrupt disconnect) never
    crashes the server or affects other connections.
    """
    manager: ConnectionManager = get_connection_manager()
    client_id = str(uuid.uuid4())

    await manager.connect(websocket, client_id)
    logger.info("WebSocket client connected", extra={"client_id": client_id})

    try:
        await websocket.send_json(build_connected_message(client_id))
        while True:
            raw_text = await websocket.receive_text()
            response = create_response_for_message(raw_text)
            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected", extra={"client_id": client_id})
    except RuntimeError as exc:
        # Starlette raises RuntimeError when the peer disappears mid-read/send.
        logger.info(
            "WebSocket client went away",
            extra={"client_id": client_id, "error": str(exc)},
        )
    except Exception as exc:
        logger.error(
            "Unexpected error in WebSocket handler",
            extra={"client_id": client_id, "error": str(exc)},
            exc_info=True,
        )
        await _try_send_error(websocket, client_id)
    finally:
        await manager.disconnect(client_id)
        logger.info("WebSocket connection closed", extra={"client_id": client_id})


async def _try_send_error(websocket: WebSocket, client_id: str) -> None:
    """Best-effort delivery of a generic server error; never raises.

    Args:
        websocket: The connection to notify.
        client_id: Id of the affected client (for logging only).
    """
    try:
        await websocket.send_json(
            build_error_message(
                ERROR_INTERNAL, "An unexpected server error occurred."
            )
        )
    except Exception:
        logger.warning(
            "Unable to notify WebSocket client of a server error",
            extra={"client_id": client_id},
        )
