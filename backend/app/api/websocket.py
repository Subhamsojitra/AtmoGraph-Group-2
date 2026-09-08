"""WebSocket endpoint for AtmoGraph (Modules 15, 16 & 17).

Module 15 provides the WebSocket transport foundation; Module 16 connects it
to the existing GNN prediction pipeline; Module 17 adds the ripple-effect
prediction streaming lifecycle. This module implements transport concerns only:

    * connection lifecycle (accept -> connected -> loop -> disconnect)
    * JSON receive / send
    * Pydantic-driven message validation
    * structured, client-safe errors (no stack traces / paths / credentials)

The actual ML work is delegated to the existing :class:`PredictionService`
through :func:`app.services.websocket_prediction.run_websocket_prediction` and
to the Module 17 :class:`RipplePredictionService` through
:func:`run_ripple_prediction`, so this module contains NO ML logic:
establishing a connection, pinging, receiving pong and validating request
payloads never touch the GNN model or the graph database. Only a validated
``prediction_request`` or ``ripple_prediction`` triggers the delegated,
thread-pooled inference / propagation pipeline.

Error codes (see :mod:`app.schemas.websocket`):

    INVALID_JSON                 unparseable JSON text
    INVALID_MESSAGE              not an object / schema violation
    UNSUPPORTED_MESSAGE_TYPE     unknown message type
    MODEL_UNAVAILABLE            no trained GNN model configured / found
    PREDICTION_FAILED            graph data, inference or unexpected failure
    NODE_NOT_FOUND               requested node id has no prediction
    INTERNAL_ERROR               unexpected server-side failure
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.logger import get_logger
from app.schemas.websocket import (
    ERROR_INTERNAL,
    ERROR_INVALID_JSON,
    ERROR_INVALID_MESSAGE,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_NOT_SUPPORTED_YET,
    ERROR_PREDICTION_FAILED,
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
    build_ripple_completed_message,
    build_ripple_detected_message,
    build_ripple_prediction_result_message,
    build_ripple_progress_message,
    build_ripple_started_message,
    summarize_validation_error,
)
from app.services.websocket_manager import (
    ConnectionManager,
    get_connection_manager,
)
from app.services.websocket_prediction import (
    ModelNotAvailableError,
    PredictionService,
    get_websocket_prediction_service,
    run_websocket_prediction,
)
from app.services.ripple_prediction.ripple_prediction_service import (
    RipplePredictionService,
)
from app.services.ripple_prediction.result_schema import RipplePredictionResult
from app.services.ripple_prediction.exceptions import (
    EntityNotFoundError,
    GNNPredictionError,
    RipplePredictionError,
    ServiceUnavailable,
)

# --------------------------------------------------------------------------- #
# Dependency: RipplePredictionService (singleton, reused across connections)
# --------------------------------------------------------------------------- #

_ripple_prediction_service: Optional[RipplePredictionService] = None


def get_ripple_prediction_service() -> RipplePredictionService:
    """Provide the shared :class:`RipplePredictionService`.

    A single service instance is reused across connections so the underlying
    GraphService / RiskPropagationService / PredictionService (and their
    Neo4j connections) are not duplicated per client.
    """
    global _ripple_prediction_service
    if _ripple_prediction_service is None:
        _ripple_prediction_service = RipplePredictionService()
    return _ripple_prediction_service


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
    # ripple-effect processing is delegated to the route asynchronously:
    # create_response_for_message stays pure and synchronous.
    if message.type == MESSAGE_TYPE_RIPPLE_PREDICTION_REQUEST:
        try:
            WebSocketRipplePredictionRequest(**(message.data or {}))
        except (ValidationError, TypeError) as exc:
            return build_error_message(
                ERROR_INVALID_MESSAGE, _summarize_validation_error(exc)
            )
        return message

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
# Ripple prediction dispatcher (Module 17, Part 4)
# --------------------------------------------------------------------------- #

_ENTITY_NOT_FOUND_MESSAGE = (
    "Ripple prediction failed: the requested entity was not found in the graph."
)
_NEO4J_UNAVAILABLE_MESSAGE = (
    "Ripple prediction failed: the graph database is currently unavailable."
)
_PREDICTION_UNAVAILABLE_MESSAGE = (
    "Ripple prediction failed: GNN prediction is currently unavailable."
)
_RIPPLE_PREDICTION_FAILED_MESSAGE = (
    "Ripple prediction failed due to a server error."
)


async def run_ripple_prediction(
    message: InboundWebSocketMessage,
    ripple_service: RipplePredictionService,
) -> AsyncIterator[dict[str, Any]]:
    """Stream a ripple prediction request through the Module 17 service.

    Pure orchestration: validates the payload, emits the streaming lifecycle
    events, delegates the business logic to :class:`RipplePredictionService`,
    and maps every failure to a structured, client-safe WebSocket error. No ML
    / graph / propagation logic lives here.

    The blocking service call is moved off the event loop (via
    ``run_in_threadpool``) so the WebSocket remains responsive to other
    clients while the prediction runs. Each lifecycle event is yielded as soon
    as it is produced, enabling real-time streaming:

        ripple_prediction_started
        ripple_prediction_progress  (risk_propagation stage)
        ripple_detected             (one per REAL affected entity)
        ripple_prediction_progress  (gnn_prediction stage)
        ripple_prediction_result
        ripple_prediction_completed

    Args:
        message: A validated ``InboundWebSocketMessage`` of type
            ``ripple_prediction``.
        ripple_service: The Module 17 ripple prediction service.

    Yields:
        JSON-serializable envelopes: the streaming lifecycle events above,
        or a structured ``error`` envelope (on any expected failure). The
        WebSocket connection is never terminated by a failing request.
    """
    request = _build_ripple_request(message)
    if isinstance(request, dict):
        # Validation failed; request is already an error envelope.
        yield request
        return

    # 1. Signal that processing has started with the resolved source info.
    yield build_ripple_started_message(
        entity_id=request.entity_id,
        entity_name=request.entity_name,
        risk_score=request.risk_score,
    )

    # 2. Emit a stage-based progress event BEFORE delegating the blocking
    #    service call to a worker thread so other clients stay responsive.
    yield build_ripple_progress_message(
        stage="risk_propagation",
        message="Propagating risk through the supply chain",
    )

    try:
        result: RipplePredictionResult = await run_in_threadpool(
            ripple_service.predict, request
        )
    except EntityNotFoundError as exc:
        logger.info(
            "WebSocket ripple prediction: entity not found",
            extra={"error": str(exc)},
        )
        yield build_error_message(
            ERROR_INVALID_MESSAGE, _ENTITY_NOT_FOUND_MESSAGE
        )
        return
    except ServiceUnavailable as exc:
        logger.error(
            "WebSocket ripple prediction: Neo4j unavailable",
            extra={"error": str(exc)},
        )
        yield build_error_message(
            ERROR_PREDICTION_FAILED, _NEO4J_UNAVAILABLE_MESSAGE
        )
        return
    except ModelNotAvailableError as exc:
        logger.warning(
            "WebSocket ripple prediction: GNN model not available",
            extra={"error": str(exc)},
        )
        yield build_error_message(
            ERROR_MODEL_UNAVAILABLE, _PREDICTION_UNAVAILABLE_MESSAGE
        )
        return
    except GNNPredictionError as exc:
        logger.error(
            "WebSocket ripple prediction: GNN inference failed",
            extra={"error": str(exc)},
        )
        yield build_error_message(
            ERROR_PREDICTION_FAILED, _RIPPLE_PREDICTION_FAILED_MESSAGE
        )
        return
    except RipplePredictionError as exc:
        logger.error(
            "WebSocket ripple prediction: service failure",
            extra={"error": str(exc)},
        )
        yield build_error_message(
            ERROR_PREDICTION_FAILED, _RIPPLE_PREDICTION_FAILED_MESSAGE
        )
        return
    except Exception:
        logger.error(
            "WebSocket ripple prediction: unexpected failure", exc_info=True
        )
        yield build_error_message(
            ERROR_PREDICTION_FAILED, _RIPPLE_PREDICTION_FAILED_MESSAGE
        )
        return

    # 3. Emit ripple_detected for each REAL affected entity. Zero affected
    #    entities means zero ripple_detected events — nothing is invented.
    for entity in result.affected_entities:
        yield build_ripple_detected_message(entity)

    # 4. Emit a progress event reflecting that GNN prediction is done.
    yield build_ripple_progress_message(
        stage="gnn_prediction",
        message="Enriching affected entities with GNN predictions",
        affected_count=result.affected_count,
    )

    # 5. Emit the final structured result.
    try:
        yield build_ripple_prediction_result_message(result)
    except Exception:
        logger.error(
            "WebSocket ripple prediction: response serialization failed",
            exc_info=True,
        )
        yield build_error_message(
            ERROR_PREDICTION_FAILED, _RIPPLE_PREDICTION_FAILED_MESSAGE
        )
        return

    # 6. Signal successful completion of the lifecycle.
    yield build_ripple_completed_message(
        source_entity_id=result.source_entity_id,
        affected_count=result.affected_count,
        prediction_count=result.prediction_count,
    )


def _build_ripple_request(
    message: InboundWebSocketMessage,
) -> WebSocketRipplePredictionRequest | dict[str, Any]:
    """Validate the message payload into a :class:`WebSocketRipplePredictionRequest`.

    Returns a structured ``error`` envelope (as a dict) when the payload is
    invalid; the route already validated it, so this is defense in depth.
    """
    data = message.data or {}
    if not isinstance(data, dict):
        return build_error_message(
            ERROR_INVALID_MESSAGE,
            "The 'data' payload of a ripple_prediction must be a JSON object.",
        )
    try:
        return WebSocketRipplePredictionRequest(**data)
    except ValidationError as exc:
        return build_error_message(
            ERROR_INVALID_MESSAGE, summarize_validation_error(exc)
        )
    except (TypeError, ValueError) as exc:
        logger.warning(
            "WebSocket ripple prediction: invalid payload",
            extra={"error": str(exc)},
        )
        return build_error_message(
            ERROR_INVALID_MESSAGE, "Invalid ripple prediction payload."
        )


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #


@router.websocket(WEBSOCKET_ROUTE_PATH)
async def websocket_endpoint(
    websocket: WebSocket,
    prediction_service: PredictionService = Depends(
        get_websocket_prediction_service
    ),
    ripple_service: RipplePredictionService = Depends(
        get_ripple_prediction_service
    ),
) -> None:
    """Serve the WebSocket transport at ``/api/v1/ws`` (Modules 15, 16 & 17).

    Lifecycle:

        1. accept the connection and register it with the connection manager
        2. send the ``connected`` message (client_id + protocol info)
        3. loop: receive one text frame -> validate -> respond
           - ``ping`` is answered synchronously with ``pong``
           - a validated ``prediction_request`` triggers the (thread-pooled)
             Module 14 inference through the delegated dispatcher
           - a validated ``ripple_prediction`` triggers the Module 17
             ripple-effect pipeline through the delegated dispatcher
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
                if response.type == MESSAGE_TYPE_RIPPLE_PREDICTION_REQUEST:
                    # Valid ripple_prediction: stream the Module 17
                    # ripple-effect lifecycle events. Each event (started,
                    # progress, detected, result, completed) is sent to the
                    # client as soon as it is produced. The stream is
                    # self-terminating, so no further send is needed here.
                    async for event in run_ripple_prediction(
                        response, ripple_service
                    ):
                        await websocket.send_json(event)
                elif response.type == MESSAGE_TYPE_PREDICTION_REQUEST:
                    # Valid prediction_request: run the existing prediction
                    # pipeline (blocking inference inside a worker thread)
                    # and send either the prediction_result or a structured
                    # error.
                    response = await run_websocket_prediction(
                        response, prediction_service
                    )
                    await websocket.send_json(response)
                else:
                    # Future InboundWebSocketMessage types: raise a clear
                    # error if a new type is added without a handler.
                    raise ValueError(
                        f"Unhandled InboundWebSocketMessage type: "
                        f"{response.type!r}"
                    )
            else:
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
