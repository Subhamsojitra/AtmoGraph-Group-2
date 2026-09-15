"""Domain exceptions for the ripple prediction service (Module 17).

The WebSocket layer maps these exceptions to structured, client-safe error
responses. No exception here exposes stack traces, filesystem paths or
credentials.
"""

from __future__ import annotations

from app.ml.exceptions import GNNPredictionError  # re-exported for transport layer
from neo4j.exceptions import ServiceUnavailable  # re-exported for transport layer
from app.services.risk.exceptions import EntityNotFoundError  # re-exported for transport layer


class RipplePredictionError(Exception):
    """Base class for all ripple prediction failures.

    The service raises controlled, typed exceptions so the WebSocket layer
    can map each failure category to a stable error code without parsing
    messages.
    """


__all__ = [
    "EntityNotFoundError",
    "GNNPredictionError",
    "RipplePredictionError",
    "ServiceUnavailable",
]
