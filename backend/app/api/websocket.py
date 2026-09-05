"""WebSocket endpoint for AtmoGraph (Modules 15 & 16).

Module 15 provides the WebSocket transport foundation; Module 16 connects it
to the existing GNN prediction pipeline. This module implements transport
concerns only:

    * connection lifecycle (accept -> connected -> loop -> disconnect)
    * JSON receive / send
    * Pydantic-driven message validation
    * structured, client-safe errors (no stack traces / paths / credentials)

The actual ML work is delegated to the existing :class:`PredictionService`
through :func:`app.services.websocket_prediction.run_websocket_prediction`, so
this module contains NO ML logic: establishing a connection, pinging, receiving
pong and validating prediction_request payloads never touch the GNN model.
Only a validated ``prediction_request`` triggers the (thread-pooled) Module 14
inference.

Error codes (see :mod:`app.schemas.websocket`):

    INVALID_JSON                 unparseable JSON text
    INVALID_MESSAGE              not an object / schema violation
    UNSUPPORTED_MESSAGE_TYPE     unknown message type
    NOT_SUPPORTED_YET            reserved for Module 17 (ripple_prediction)
    MODEL_UNAVAILABLE            no trained GNN model configured / found
    PREDICTION_FAILED            graph data, inference or unexpected failure
    NODE_NOT_FOUND               requested node id has no prediction
    INTERNAL_ERROR               unexpected server-side failure
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.logger import get_logger
from app.schemas.websocket import (
    ERROR_INTERNAL,
    ERROR_INVALID_JSON,
    ERROR_INVALID_MESSAGE,
    ERROR_NOT_SUPPORTED_YET,
    ERROR_UNSUPPORTED_MESSAGE_TYPE,
    MESSAGE_TYPE_PING,
    MESSAGE_TYPE_PREDICTION_REQUEST,
    MESSAGE_TYPE_RIPPLE_PREDICTION_REQUEST,
    RESERVED_CLIENT_MESSAGE_TYPES,
    SUPPORTED_CLIENT_MESSAGE_TYPES,
    InboundWebSocketMessage,
    WebSocketPredictionRequest,
    WebSocketRipplePredictionRequest,
    build_connected_message,
    build_error_message,
    build_pong_message,
    summarize_validation_error,
)
from app.services.websocket_manager import (
    ConnectionManager,
    get_connection_manager,
)
from app.services.websocket_prediction import (
    PredictionService,
    get_websocket_prediction_service,
    run_websocket_prediction,
)

logger = get_logger(__name__)

router = APIRouter(tags=["websocket"])

#: Route path (relative to the ``/api/v1`` prefix applied in ``app.main``).
#: The full endpoint is ``/api/v1/ws``.
WEBSOCKET_ROUTE_PATH = "/ws"


# --------------------------------------------------------------------------- #
# Message processing (pure, socket-free helpers - unit-testable)
# --------------------------------------------------------------------------- #


def create_response_for_message(
    raw_text: str,
) -> dict[str, Any] | InboundWebSocketMessage:
    """Validate one raw client message and return the envelope to send back.

    Pure, network-free and ML-free. It never raises, so an endpoint loop can
    pipe any client input into it and always receive either a ready-to-send
    response envelope or, for a structurally VALID ``prediction_request``, the
    validated message itself — the (blocking, thread-pooled) Module 14
    inference is run afterwards by the async route handler, never here.

    Args:
        raw_text: The raw text frame received from the client.

    Returns:
        A JSON-serializable envelope (``pong`` or ``error``) or, for a valid
        ``prediction_request``, the validated :class:`InboundWebSocketMessage`.
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

    # A prediction_request is validated here (cheap) but INFERRED asynchronously
    # by the route: create_response_for_message stays pure and synchronous.
    if message.type == MESSAGE_TYPE_PREDICTION_REQUEST:
        try:
            WebSocketPredictionRequest(**(message.data or {}))
        except (ValidationError, TypeError) as exc:
            return build_error_message(
                ERROR_INVALID_MESSAGE, _summarize_validation_error(exc)
            )
        return message

    # A ripple_prediction request is validated here (Module 17). The actual
    # ripple-effect processing is not implemented yet, so a valid request is
    # answered with NOT_SUPPORTED_YET; an invalid payload is rejected with a
    # structured INVALID_MESSAGE error. This keeps the transport pure and
    # synchronous while enforcing the Module 17 request contract.
    if message.type == MESSAGE_TYPE_RIPPLE_PREDICTION_REQUEST:
        try:
            WebSocketRipplePredictionRequest(**(message.data or {}))
        except (ValidationError, TypeError) as exc:
            return build_error_message(
                ERROR_INVALID_MESSAGE, _summarize_validation_error(exc)
            )
        return build_error_message(
            ERROR_NOT_SUPPORTED_YET,
            "Ripple prediction is recognized but the Module 17 "
            "ripple-effect streaming pipeline is not implemented yet.",
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
    """Dispatch a validated NON-prediction message to its Module 15 handler.

    ``prediction_request`` never reaches this function — it is intercepted by
    :func:`create_response_for_message` and handled asynchronously by the
    route. This function serves ``pong`` replies and structured errors for the
    remaining (unknown / reserved) message types.

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
            f"Message type '{message_type}' is reserved for the Module 17 "
            "ripple-effect streaming pipeline and is not handled yet.",
        )
    supported = ", ".join(sorted(SUPPORTED_CLIENT_MESSAGE_TYPES))
    return build_error_message(
        ERROR_UNSUPPORTED_MESSAGE_TYPE,
        f"Unsupported message type '{message_type}'. "
        f"Supported types: {supported}.",
    )


def _summarize_validation_error(exc: Exception) -> str:
    """Summarize a Pydantic validation error into client-safe text.

    Non-Pydantic structural failures (e.g. ``TypeError`` from the model
    constructor) fall back to a generic message.

    Args:
        exc: The exception raised while validating the inbound message.

    Returns:
        A one-line description of the first failing field when available.
    """
    if isinstance(exc, ValidationError):
        return summarize_validation_error(exc)
    return "Invalid message."


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #


@router.websocket(WEBSOCKET_ROUTE_PATH)
async def websocket_endpoint(
    websocket: WebSocket,
    prediction_service: PredictionService = Depends(
        get_websocket_prediction_service
    ),
) -> None:
    """Serve the WebSocket transport at ``/api/v1/ws`` (Modules 15 & 16).

    Lifecycle:

        1. accept the connection and register it with the connection manager
        2. send the ``connected`` message (client_id + protocol info)
        3. loop: receive one text frame -> validate -> respond
           - ``ping`` is answered synchronously with ``pong``
           - a validated ``prediction_request`` triggers the (thread-pooled)
             Module 14 inference through the delegated dispatcher
        4. on disconnect / unexpected failure: remove the client, then close

    One faulty client (bad JSON, unknown type, invalid prediction request,
    failing inference, abrupt disconnect) never crashes the server or affects
    other connections, and an invalid prediction request never terminates the
    connection that sent it.
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
            if isinstance(response, InboundWebSocketMessage):
                # Valid prediction_request: run the existing prediction
                # pipeline (blocking inference inside a worker thread) and
                # send either the prediction_result or a structured error.
                response = await run_websocket_prediction(
                    response, prediction_service
                )
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
