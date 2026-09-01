"""Pydantic schemas for the Module 15 WebSocket transport.

This module defines the minimal, stable wire contract used by the
``/api/v1/ws`` WebSocket endpoint. Module 15 only implements connection
lifecycle and the ``ping``/``pong`` handshake; the messages for the ML /
ripple-prediction streaming pipeline (``prediction_request`` /
``ripple_prediction``) are *reserved* here so Modules 16/17 can adopt them
without changing the transport.

Inbound (client -> server):

    {"type": "ping"}

Outbound (server -> client):

    {"type": "connected", "data": {"client_id": ..., "protocol": ...}}
    {"type": "pong", "data": {"echo": ...}}
    {"type": "error", "error": {"code": ..., "message": ...}}

Validation is strict on purpose: unknown top-level fields are rejected (rather
than silently ignored) so the transport never accepts arbitrary structures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Message type constants (Module 15 transport contract)
# --------------------------------------------------------------------------- #

MESSAGE_TYPE_PING = "ping"
MESSAGE_TYPE_CONNECTED = "connected"
MESSAGE_TYPE_PONG = "pong"
MESSAGE_TYPE_ERROR = "error"

#: Transport protocol version advertised in the ``connected`` message. Bump it
#: whenever the wire contract changes incompatibly.
PROTOCOL_VERSION = 1

#: Client -> server message types handled by Module 15.
SUPPORTED_CLIENT_MESSAGE_TYPES: frozenset[str] = frozenset({MESSAGE_TYPE_PING})

#: Message types reserved for the Module 16/17 ML streaming pipeline. They are
#: recognized so the client receives a precise "not supported yet" error
#: instead of a generic one, but they are NOT handled in Module 15.
RESERVED_CLIENT_MESSAGE_TYPES: frozenset[str] = frozenset(
    {"prediction_request", "ripple_prediction"}
)

# --------------------------------------------------------------------------- #
# Structured error codes (stable, machine-readable)
# --------------------------------------------------------------------------- #

ERROR_INVALID_JSON = "INVALID_JSON"
ERROR_INVALID_MESSAGE = "INVALID_MESSAGE"
ERROR_UNSUPPORTED_MESSAGE_TYPE = "UNSUPPORTED_MESSAGE_TYPE"
ERROR_NOT_SUPPORTED_YET = "NOT_SUPPORTED_YET"
ERROR_INTERNAL = "INTERNAL_ERROR"


def _utc_now() -> datetime:
    """Return the current UTC time (default for envelope timestamps)."""
    return datetime.now(timezone.utc)


class InboundWebSocketMessage(BaseModel):
    """A validated JSON message received from a WebSocket client.

    Every inbound message must be a JSON object carrying a non-empty string
    ``type``. ``data`` is optional and reserved for module-specific payloads;
    Module 15 accepts only the ``ping`` type (see
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
    transport/module payloads for ``connected``/``pong`` messages.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["connected", "pong", "error"]
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