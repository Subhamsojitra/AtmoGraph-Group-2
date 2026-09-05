"""Pydantic schemas for the WebSocket transport (Modules 15 & 16).

This module defines the minimal, stable wire contract used by the
``/api/v1/ws`` WebSocket endpoint.

Module 15 implements the connection lifecycle and the ``ping``/``pong``
handshake. Module 16 adds the ML prediction streaming messages
(``prediction_request`` / ``prediction_result``) on top of that unchanged
transport. ``ripple_prediction`` remains a *reserved* Module 17 message.

Inbound (client -> server):

    {"type": "ping"}
    {"type": "prediction_request", "data": {"node_id": ..., "relationship_types": [...]}}

Outbound (server -> client):

    {"type": "connected", "data": {"client_id": ..., "protocol": ...}}
    {"type": "pong", "data": {"echo": ...}}
    {"type": "prediction_result", "data": {... PredictionResponse ...}}
    {"type": "error", "error": {"code": ..., "message": ...}}

Validation is strict on purpose: unknown top-level fields are rejected (rather
than silently ignored) so the transport never accepts arbitrary structures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.schemas.prediction import PredictionRequest, PredictionResponse

# --------------------------------------------------------------------------- #
# Message type constants (WebSocket transport contract)
# --------------------------------------------------------------------------- #

MESSAGE_TYPE_PING = "ping"
MESSAGE_TYPE_CONNECTED = "connected"
MESSAGE_TYPE_PONG = "pong"
MESSAGE_TYPE_ERROR = "error"
MESSAGE_TYPE_PREDICTION_REQUEST = "prediction_request"
MESSAGE_TYPE_PREDICTION_RESULT = "prediction_result"

#: Module 17 ripple-effect streaming message types.
MESSAGE_TYPE_RIPPLE_PREDICTION_REQUEST = "ripple_prediction"
MESSAGE_TYPE_RIPPLE_PREDICTION_STARTED = "ripple_prediction_started"
MESSAGE_TYPE_RIPPLE_PREDICTION_PROGRESS = "ripple_prediction_progress"
MESSAGE_TYPE_RIPPLE_DETECTED = "ripple_detected"
MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT = "ripple_prediction_result"
MESSAGE_TYPE_RIPPLE_PREDICTION_COMPLETED = "ripple_prediction_completed"

#: Transport protocol version advertised in the ``connected`` message. Bump it
#: whenever the wire contract changes incompatibly.
PROTOCOL_VERSION = 1

#: Client -> server message types handled by the transport.
SUPPORTED_CLIENT_MESSAGE_TYPES: frozenset[str] = frozenset(
    {
        MESSAGE_TYPE_PING,
        MESSAGE_TYPE_PREDICTION_REQUEST,
        MESSAGE_TYPE_RIPPLE_PREDICTION_REQUEST,
    }
)

#: Message types reserved for future extensions (no longer used by Module 17).
#: Kept for backward compatibility with clients that may check for reserved types.
RESERVED_CLIENT_MESSAGE_TYPES: frozenset[str] = frozenset()

# --------------------------------------------------------------------------- #
# Structured error codes (stable, machine-readable)
# --------------------------------------------------------------------------- #

ERROR_INVALID_JSON = "INVALID_JSON"
ERROR_INVALID_MESSAGE = "INVALID_MESSAGE"
ERROR_UNSUPPORTED_MESSAGE_TYPE = "UNSUPPORTED_MESSAGE_TYPE"
ERROR_NOT_SUPPORTED_YET = "NOT_SUPPORTED_YET"
ERROR_INTERNAL = "INTERNAL_ERROR"

#: Module 16 prediction error codes (same ``error`` envelope, distinct codes).
ERROR_MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
ERROR_PREDICTION_FAILED = "PREDICTION_FAILED"
ERROR_NODE_NOT_FOUND = "NODE_NOT_FOUND"


def _utc_now() -> datetime:
    """Return the current UTC time (default for envelope timestamps)."""
    return datetime.now(timezone.utc)


class InboundWebSocketMessage(BaseModel):
    """A validated JSON message received from a WebSocket client.

    Every inbound message must be a JSON object carrying a non-empty string
    ``type``. ``data`` is optional and reserved for module-specific payloads;
    ``ping`` and ``prediction_request`` are handled by Modules 15/16 (see
    :data:`SUPPORTED_CLIENT_MESSAGE_TYPES`).
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        min_length=1,
        max_length=50,
        description="Message type identifier, e.g. 'ping'",
    )
    data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional module-specific payload; must be a JSON object",
    )

    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        """Trim ``type`` and reject whitespace-only values."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("'type' must be a non-empty string")
        return stripped


class WebSocketPredictionRequest(PredictionRequest):
    """Prediction request payload for the ``prediction_request`` message.

    Extends the Module 14 :class:`~app.schemas.prediction.PredictionRequest`
    (which only knows ``relationship_types``) with an OPTIONAL ``node_id`` so a
    client can ask for a single node's prediction. The pipeline itself remains
    the existing whole-graph Module 14 inference; an optional ``node_id`` only
    selects that node from the REAL model output.

    Unlike the REST schema (which ignores unknown fields), this schema REJECTS
    unknown fields so a typo such as the frontend-style ``nodeId`` cannot be
    silently ignored — the client gets a precise ``INVALID_MESSAGE`` error
    instead.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional graph node id whose prediction should be returned. "
            "Omitted = predictions for every graph node (same as the REST API)."
        ),
    )

    @field_validator("node_id")
    @classmethod
    def clean_node_id(cls, value: Optional[str]) -> Optional[str]:
        """Trim ``node_id``; explicit blank/non-string values are rejected.

        ``None`` (field omitted) means "predict for the whole graph".
        """
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("node_id must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("node_id must not be blank")
        return stripped


class WebSocketRipplePredictionRequest(BaseModel):
    """Request payload for the ``ripple_prediction`` message (Module 17).

    Combines risk propagation parameters with GNN prediction to compute the
    ripple effect of a source entity's disruption through the supply chain.
    The source entity's risk score is propagated downstream using the existing
    Module 10 risk propagation logic, and the GNN prediction is run for the
    affected entities.

    ``entity_id`` is the Module 8 ``node_id`` for a *matched* entity. A missing,
    null, or blank value means the entity is unresolved, in which case the
    service returns a controlled error and never reads the database.

    ``risk_score`` is optional; when omitted, the entity's persisted risk score
    from Module 9 is used. When provided, it must be within ``[0, 100]``.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description=(
            "Resolved entity identifier (Module 8 node_id). Blank/null means "
            "the entity is unresolved and no ripple prediction is possible."
        ),
    )
    entity_name: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Resolved entity display name (Module 8 node_name)",
    )
    risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description=(
            "Risk score of the source entity on the 0-100 scale. When omitted, "
            "the entity's persisted risk score from Module 9 is used."
        ),
    )
    max_depth: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum traversal depth (hops). Defaults to the configured value.",
    )
    attenuation: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Per-hop attenuation factor applied to the source risk score. "
            "Defaults to the configured value."
        ),
    )
    relationship_types: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional relationship type filters to follow downstream. When "
            "omitted, all outgoing relationship types are traversed."
        ),
    )

    @field_validator("entity_id")
    @classmethod
    def normalize_entity_id(cls, value: Optional[str]) -> Optional[str]:
        """Normalize ``entity_id``: whitespace-only values become ``None``."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("entity_name")
    @classmethod
    def normalize_entity_name(cls, value: Optional[str]) -> Optional[str]:
        """Normalize ``entity_name``: whitespace-only values become ``None``."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("relationship_types")
    @classmethod
    def clean_relationship_types(
        cls, value: Optional[list[str]]
    ) -> Optional[list[str]]:
        """Drop blank entries and blank/whitespace-only relationship types."""
        if value is None:
            return None
        cleaned = [
            t.strip()
            for t in value
            if isinstance(t, str) and t.strip()
        ]
        return cleaned or None


def summarize_validation_error(exc: ValidationError) -> str:
    """Turn a Pydantic :class:`ValidationError` into a short client-safe text.

    Args:
        exc: The validation error raised for an inbound message or payload.

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


class WebSocketErrorPayload(BaseModel):
    """Structured error attached to every ``error`` response."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        description="Stable machine-readable error code, e.g. 'INVALID_JSON'"
    )
    message: str = Field(
        description="Short, client-safe description of the problem"
    )


class OutboundWebSocketMessage(BaseModel):
    """Envelope for every message the server sends to a WebSocket client.

    ``error`` is only populated when ``type == "error"``; ``data`` carries
    transport/module payloads for ``connected``/``pong``/``prediction_result``
    messages.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "connected",
        "pong",
        "error",
        "prediction_result",
        "ripple_prediction_started",
        "ripple_prediction_progress",
        "ripple_detected",
        "ripple_prediction_result",
        "ripple_prediction_completed",
    ]
    timestamp: datetime = Field(default_factory=_utc_now)
    data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Transport or module-specific payload",
    )
    error: Optional[WebSocketErrorPayload] = Field(
        default=None,
        description="Structured error; present only when type == 'error'",
    )


# --------------------------------------------------------------------------- #
# Response builders (kept in the schema module so the wire shape lives in one
# place and is directly testable)
# --------------------------------------------------------------------------- #


def build_connected_message(client_id: str) -> dict[str, Any]:
    """Build the ``connected`` message sent immediately after acceptance.

    Args:
        client_id: Server-assigned identifier for this connection (later
            modules can use it to address individual clients).

    Returns:
        A JSON-serializable ``connected`` envelope.
    """
    return OutboundWebSocketMessage(
        type=MESSAGE_TYPE_CONNECTED,
        data={
            "client_id": client_id,
            "protocol": PROTOCOL_VERSION,
            "supported_client_messages": sorted(SUPPORTED_CLIENT_MESSAGE_TYPES),
        },
    ).model_dump(mode="json", exclude_none=True)


def build_pong_message(data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Build the ``pong`` reply to a validated ``ping`` message.

    When the client attached a ``data`` object to its ping, it is echoed back
    so round trips can be traced.

    Args:
        data: The ``data`` payload from the validated ping (may be ``None``).

    Returns:
        A JSON-serializable ``pong`` envelope.
    """
    payload: dict[str, Any] = {"type": MESSAGE_TYPE_PONG}
    if data is not None:
        payload["data"] = {"echo": data}
    return OutboundWebSocketMessage(**payload).model_dump(
        mode="json", exclude_none=True
    )


def build_error_message(code: str, message: str) -> dict[str, Any]:
    """Build a structured ``error`` envelope.

    Errors never contain stack traces, filesystem paths or credentials.

    Args:
        code: Stable error code (see the ``ERROR_*`` constants).
        message: Short, client-safe error description.

    Returns:
        A JSON-serializable ``error`` envelope.
    """
    return OutboundWebSocketMessage(
        type=MESSAGE_TYPE_ERROR,
        error=WebSocketErrorPayload(code=code, message=message),
    ).model_dump(mode="json", exclude_none=True)


def build_prediction_result_message(
    response: PredictionResponse,
    requested_node_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build the ``prediction_result`` envelope for a validated prediction.

    The ``data`` payload IS the existing Module 14 :class:`PredictionResponse`
    (``predictions`` / ``prediction_count`` / ``timestamp``), so the WebSocket
    wire shape reuses the REST contract instead of inventing a parallel one.
    Each entry exposes the graph ``node_id`` that the frontend maps to its
    graph node id (``prediction.nodeId``).

    When a single node was requested, its id is echoed as
    ``data.requested_node_id`` so clients can correlate the response without
    scanning the list.

    Args:
        response: A :class:`PredictionResponse` produced by the Module 14
            service (or a single-node subset of it). Only real model output is
            ever serialized — no values are invented here.
        requested_node_id: The graph node id the client asked for, when a
            single node was requested.

    Returns:
        A JSON-serializable ``prediction_result`` envelope.
    """
    payload: dict[str, Any] = {
        "type": MESSAGE_TYPE_PREDICTION_RESULT,
        "data": response.model_dump(mode="json"),
    }
    if requested_node_id is not None:
        payload["data"]["requested_node_id"] = requested_node_id
    return OutboundWebSocketMessage(**payload).model_dump(
        mode="json", exclude_none=True
    )


# --------------------------------------------------------------------------- #
# Module 17: Ripple-effect prediction builders
# --------------------------------------------------------------------------- #


def build_ripple_started_message(
    entity_id: str,
    entity_name: Optional[str] = None,
    risk_score: Optional[float] = None,
) -> dict[str, Any]:
    """Build the ``ripple_prediction_started`` message (Module 17).

    Sent when a valid ripple prediction request is accepted and processing begins.

    Args:
        entity_id: The source entity identifier.
        entity_name: The source entity display name (if available).
        risk_score: The source risk score used for propagation (if available).

    Returns:
        A JSON-serializable ``ripple_prediction_started`` envelope.
    """
    data: dict[str, Any] = {"entity_id": entity_id}
    if entity_name is not None:
        data["entity_name"] = entity_name
    if risk_score is not None:
        data["risk_score"] = risk_score
    return OutboundWebSocketMessage(
        type=MESSAGE_TYPE_RIPPLE_PREDICTION_STARTED, data=data
    ).model_dump(mode="json", exclude_none=True)


def build_ripple_progress_message(
    stage: str,
    message: str,
    affected_count: Optional[int] = None,
) -> dict[str, Any]:
    """Build a ``ripple_prediction_progress`` message (Module 17).

    Sent during processing to indicate actual progress stages.

    Args:
        stage: The current processing stage (e.g. "risk_propagation", "gnn_prediction").
        message: A human-readable description of the current stage.
        affected_count: Number of affected entities found so far (if applicable).

    Returns:
        A JSON-serializable ``ripple_prediction_progress`` envelope.
    """
    data: dict[str, Any] = {"stage": stage, "message": message}
    if affected_count is not None:
        data["affected_count"] = affected_count
    return OutboundWebSocketMessage(
        type=MESSAGE_TYPE_RIPPLE_PREDICTION_PROGRESS, data=data
    ).model_dump(mode="json", exclude_none=True)