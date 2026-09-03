"""WebSocket prediction dispatcher for AtmoGraph (Module 16).

Bridges the Module 15 WebSocket transport and the existing Module 14
prediction pipeline WITHOUT placing ML logic inside the transport layer::

    app.api.websocket (WebSocket route)
        |   validated InboundWebSocketMessage (type == prediction_request)
        v
    this module  (run_websocket_prediction)
        |   async wrapper; blocking inference moved off the event loop
        v
    PredictionService.get_predictions(PredictionRequest)   (Module 14, existing)
        |   lazy reusable GNNPredictor -> GraphDatasetBuilder -> GNNModel
        v
    PredictionResponse
        v
    this module  (normalizes to the WebSocket response contract)
        v
    prediction_result envelope  |  structured error envelope

Responsibilities (kept deliberately narrow):

* validate the Module 16 request payload (:class:`WebSocketPredictionRequest`);
* run the blocking Module 14 inference in a worker thread (never on the event
  loop);
* map every Module 14 / ML / Neo4j failure to a structured, client-safe
  WebSocket error (no tracebacks, paths or credentials);
* select the requested node from the REAL model output (never invents values).

The wire shapes (``prediction_result`` / ``error``, node-id field,
error codes) are defined in :mod:`app.schemas.websocket`; this module only
ORCHESTRATES and never talks to a socket directly.

HORIZONS (30/60/90-day): NOT supported. ``PredictionService`` returns one raw
scalar per graph node (the Module 14 contract) with no temporal dimension, so
no horizon fields appear anywhere and no horizon values are fabricated here.
Horizon-aware prediction is a future integration requirement.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi.concurrency import run_in_threadpool
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import ValidationError

from app.api.prediction import get_prediction_service
from app.core.logger import get_logger
from app.ml.exceptions import (
    EmptyGraphError,
    GNNModelNotAvailableError,
    GNNPredictionError,
    GraphDatasetError,
)
from app.schemas.prediction import PredictionResponse
from app.schemas.websocket import (
    ERROR_INVALID_MESSAGE,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_NODE_NOT_FOUND,
    ERROR_PREDICTION_FAILED,
    InboundWebSocketMessage,
    WebSocketPredictionRequest,
    build_error_message,
    build_prediction_result_message,
    summarize_validation_error,
)
from app.services.prediction_service import ModelNotAvailableError, PredictionService

logger = get_logger(__name__)

__all__ = [
    "ModelNotAvailableError",
    "PredictionService",
    "get_websocket_prediction_service",
    "run_websocket_prediction",
]

_MODEL_NOT_CONFIGURED_MESSAGE = (
    "Prediction is unavailable: no trained GNN model is configured on the server."
)
_MODEL_FILE_MISSING_MESSAGE = (
    "Prediction is unavailable: the configured trained GNN model could not be found."
)
_GRAPH_DATA_UNAVAILABLE_MESSAGE = (
    "Prediction failed: the graph data is unavailable or incomplete."
)
_INFERENCE_FAILED_MESSAGE = (
    "Prediction failed: the model could not produce a valid prediction."
)
_UNEXPECTED_FAILURE_MESSAGE = (
    "Prediction failed due to an unexpected server error."
)


def get_websocket_prediction_service() -> PredictionService:
    """Provide the prediction service used by the WebSocket route.

    Reuses the shared process-wide singleton owned by the REST prediction API
    (Module 14, ``app.api.prediction.get_prediction_service``) so the expensive
    GNN checkpoint is loaded at most ONCE per process regardless of whether a
    prediction arrives over HTTP or WebSocket. Tests override this dependency
    with a mock; the single model is never trained or downloaded here.
    """
    return get_prediction_service()


async def run_websocket_prediction(
    message: InboundWebSocketMessage,
    service: PredictionService,
) -> dict[str, Any]:
    """Run the existing prediction pipeline for a validated ``prediction_request``.

    Args:
        message: The validated :class:`InboundWebSocketMessage` carrying the
            prediction payload in its ``data`` field.
        service: The :class:`PredictionService` instance to invoke (a real,
            lazy singleton in production; a mock in unit tests).

    Returns:
        Either a ``prediction_result`` envelope with REAL model output, or a
        structured ``error`` envelope. This function NEVER raises: one invalid
        or failing request must not crash the WebSocket handler or affect other
        clients.
    """
    request = _build_request(message)
    if isinstance(request, dict):
        return request  # already a structured error envelope

    try:
        # Module 14 get_predictions is synchronous and blocking (Neo4j graph
        # extraction + torch forward pass). Run it in a worker thread so the
        # async event loop is never blocked by inference.
        response = await run_in_threadpool(service.get_predictions, request)
    except ModelNotAvailableError:
        logger.warning("WebSocket prediction: no trained GNN model is configured")
        return build_error_message(ERROR_MODEL_UNAVAILABLE, _MODEL_NOT_CONFIGURED_MESSAGE)
    except GNNModelNotAvailableError:
        logger.warning("WebSocket prediction: configured checkpoint is missing")
        return build_error_message(ERROR_MODEL_UNAVAILABLE, _MODEL_FILE_MISSING_MESSAGE)
    except (EmptyGraphError, GraphDatasetError) as exc:
        logger.warning(
            "WebSocket prediction: graph data unavailable",
            extra={"error": str(exc)},
        )
        return build_error_message(
            ERROR_PREDICTION_FAILED, _GRAPH_DATA_UNAVAILABLE_MESSAGE
        )
    except (GNNPredictionError, ServiceUnavailable, Neo4jError) as exc:
        logger.error(
            "WebSocket prediction: inference failed",
            extra={"error": str(exc)},
        )
        return build_error_message(ERROR_PREDICTION_FAILED, _INFERENCE_FAILED_MESSAGE)
    except Exception:
        logger.error("WebSocket prediction: unexpected failure", exc_info=True)
        return build_error_message(ERROR_PREDICTION_FAILED, _UNEXPECTED_FAILURE_MESSAGE)

    try:
        return _build_result_response(response, request.node_id)
    except Exception:
        logger.error(
            "WebSocket prediction: response normalization failed", exc_info=True
        )
        return build_error_message(ERROR_PREDICTION_FAILED, _UNEXPECTED_FAILURE_MESSAGE)


def _build_request(
    message: InboundWebSocketMessage,
) -> WebSocketPredictionRequest | dict[str, Any]:
    """Validate the message payload into a :class:`WebSocketPredictionRequest`.

    Returns a structured ``error`` envelope (as a dict) when the payload is
    invalid; the route already validated it, so this is defense in depth.
    """
    data = message.data or {}
    if not isinstance(data, dict):
        return build_error_message(
            ERROR_INVALID_MESSAGE,
            "The 'data' payload of a prediction_request must be a JSON object.",
        )
    try:
        return WebSocketPredictionRequest(**data)
    except ValidationError as exc:
        return build_error_message(
            ERROR_INVALID_MESSAGE, summarize_validation_error(exc)
        )
    except (TypeError, ValueError) as exc:
        logger.warning(
            "WebSocket prediction: invalid payload", extra={"error": str(exc)}
        )
        return build_error_message(
            ERROR_INVALID_MESSAGE, "Invalid prediction request payload."
        )


def _build_result_response(
    response: PredictionResponse,
    requested_node_id: Optional[str],
) -> dict[str, Any]:
    """Normalize the Module 14 response into the WebSocket response contract.

    A whole-graph request returns every node's prediction exactly as the REST
    API would. A single-node request selects ONLY that node from the real model
    output — nothing is invented — and reports ``NODE_NOT_FOUND`` when the id
    has no prediction in the current graph.
    """
    if requested_node_id is None:
        return build_prediction_result_message(response)

    for entry in response.predictions:
        if entry.node_id == requested_node_id:
            single = PredictionResponse(
                predictions=[entry],
                prediction_count=1,
                timestamp=response.timestamp,
            )
            return build_prediction_result_message(
                single, requested_node_id=requested_node_id
            )

    logger.info(
        "WebSocket prediction: requested node not found in the current graph",
        extra={"node_id": requested_node_id},
    )
    return build_error_message(
        ERROR_NODE_NOT_FOUND,
        f"No prediction is available for node '{requested_node_id}' in the "
        "current graph.",
    )