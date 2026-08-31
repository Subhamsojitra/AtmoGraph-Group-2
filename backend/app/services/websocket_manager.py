"""In-process WebSocket connection manager (Module 15).

Keeps track of live client connections and exposes the small set of
primitives the WebSocket route needs: register, remove, targeted send and
best-effort broadcast. This is deliberately an *in-process* registry: no
cross-process pub/sub infrastructure, which is out of scope for Module 15.

Modules 16/17 (ML / ripple-prediction streaming) can reuse these primitives
to push ``prediction_request`` / ``ripple_prediction`` messages to specific
clients (by the ``client_id`` assigned in the ``connected`` handshake) or to
broadcast to every connected client.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import WebSocket

from app.core.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Registry of live WebSocket connections.

    All mutations are serialized through an ``asyncio.Lock`` so concurrent
    sends/disconnects (e.g. during shutdown) cannot interleave unsafely.
    """

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    @property
    def active_connections(self) -> int:
        """Number of currently registered client connections."""
        return len(self._connections)

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and register a new client connection.

        If a client with the same id is already registered, it is replaced
        (stale entries are never left behind).

        Args:
            websocket: The FastAPI WebSocket to accept and store.
            client_id: Server-assigned unique id for this connection.
        """
        await websocket.accept()
        async with self._lock:
            if client_id in self._connections:
                logger.warning(
                    "Replacing an existing WebSocket connection",
                    extra={"client_id": client_id},
                )
            self._connections[client_id] = websocket

    async def disconnect(self, client_id: str) -> None:
        """Remove a client from the registry.

        No-op when the client id is absent (e.g. double disconnect).

        Args:
            client_id: Id of the connection to remove.
        """
        async with self._lock:
            self._connections.pop(client_id, None)

    def is_connected(self, client_id: str) -> bool:
        """Return True when a client id is currently registered."""
        return client_id in self._connections

    async def send_json(self, client_id: str, payload: dict[str, Any]) -> bool:
        """Best-effort JSON send to a single client.

        Returns ``False`` when the client is unknown or the transport fails
        (dead/stale connection); failed connections are removed from the
        registry so they are never reused.

        Args:
            client_id: Target client id.
            payload: JSON-serializable message envelope.

        Returns:
            True when delivered, False otherwise.
        """
        async with self._lock:
            websocket = self._connections.get(client_id)
        if websocket is None:
            return False
        try:
            await websocket.send_json(payload)
            return True
        except Exception as exc:
            logger.warning(
                "Removing WebSocket connection after a send failure",
                extra={"client_id": client_id, "error": str(exc)},
            )
            await self.disconnect(client_id)
            return False

    async def broadcast_json(self, payload: dict[str, Any]) -> int:
        """Send a JSON envelope to every connected client (best-effort).

        Args:
            payload: JSON-serializable message envelope.

        Returns:
            Number of clients that received the message.
        """
        ids = list(self._connections.keys())
        delivered = 0
        for client_id in ids:
            if await self.send_json(client_id, payload):
                delivered += 1
        return delivered

    async def close_all(self) -> None:
        """Close every live connection (used on server shutdown).

        The registry is cleared first so no new sends can target a
        being-closed connection.
        """
        async with self._lock:
            sockets = list(self._connections.values())
            self._connections.clear()
        for websocket in sockets:
            try:
                await websocket.close()
            except Exception:
                logger.warning(
                    "Error while closing a WebSocket during shutdown",
                    exc_info=True,
                )


# --------------------------------------------------------------------------- #
# Dependency helper (same singleton pattern as app.api.prediction)
# --------------------------------------------------------------------------- #

_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Provide the shared :class:`ConnectionManager`.

    A single in-process manager is reused across connections so later modules
    can target clients by id; tests call :func:`reset_connection_manager`
    between runs.
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


def reset_connection_manager() -> None:
    """Drop the shared manager instance (used by tests and reload tooling).

    Any connections still registered by the old instance are abandoned;
    the endpoint always removes its own connection in a ``finally`` block.
    """
    global _connection_manager
    _connection_manager = None