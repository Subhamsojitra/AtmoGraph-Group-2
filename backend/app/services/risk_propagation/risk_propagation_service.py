"""Risk propagation / ripple effect service for AtmoGraph (Module 10).

This module propagates the risk state of a resolved entity (produced by Module 9)
downstream through the supply-chain graph to the entities that depend on it.
It reuses the existing :class:`GraphRepository` for all database access and the
existing :class:`RiskService` risk-level calculation, so it does not create a
second Neo4j connection, a second repository, or a second risk engine.

Architecture:

    FastAPI
        ↓
    Risk Propagation API (``/api/v1/risk-propagation/propagate``)
        ↓
    RiskPropagationService (this module)
        ↓
    GraphRepository (existing) | RiskService.calculate_risk_level (existing)
        ↓
    Neo4jDatabase (existing singleton)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.config import settings
from app.core.logger import get_logger
from app.repositories.graph_repository import GraphRepository
from app.schemas.risk_propagation import (
    AffectedEntity,
    RiskPropagationRequest,
    RiskPropagationResponse,
)
from app.services.risk.exceptions import EntityNotFoundError
from app.services.risk.risk_service import RiskService

logger = get_logger(__name__)


def _unwrap_node(node: Any) -> Any:
    """Unwrap a raw Neo4j record shape into the underlying node value.

    ``get_node_by_id`` returns a record dictionary shaped ``{"n": <node>}`` (or
    ``{"node": <node>}`` for traversal records). This helper returns the inner
    node value so name/score extraction can read its properties directly, while
    leaving already-unwrapped nodes untouched.
    """
    if isinstance(node, dict) and "id" not in node:
        for key in ("n", "node"):
            value = node.get(key)
            if value is not None:
                return value
    return node


class RiskPropagationService:
    """Propagate a resolved entity's risk downstream through the graph (Module 10).

    External callers (the API layer) use this service instead of accessing the
    repository or database manager directly. Business logic lives here rather
    than inside FastAPI route handlers.
    """

    def __init__(
        self,
        repository: Optional[GraphRepository] = None,
        risk_service: Optional[RiskService] = None,
        score_min: Optional[float] = None,
        score_max: Optional[float] = None,
        default_max_depth: Optional[int] = None,
        default_attenuation: Optional[float] = None,
        max_affected: Optional[int] = None,
    ) -> None:
        """Initialize a risk propagation service.

        Args:
            repository: Existing :class:`GraphRepository` to reuse. Defaults to a
                new instance bound to the existing Neo4j connection.
            risk_service: Existing :class:`RiskService` used only for deriving
                risk levels. Defaults to a new instance (its risk engine is
                reused, not duplicated).
            score_min: Lowest accepted risk score (default from settings).
            score_max: Highest accepted risk score (default from settings).
            default_max_depth: Default traversal depth when the request omits it.
            default_attenuation: Default per-hop attenuation when the request
                omits it.
            max_affected: Upper bound on the number of affected entities kept.
        """
        self._repository = repository or GraphRepository()
        self._risk_service = risk_service or RiskService()
        self._score_min = settings.risk_score_min if score_min is None else score_min
        self._score_max = settings.risk_score_max if score_max is None else score_max
        self._default_max_depth = (
            settings.risk_propagation_max_depth
            if default_max_depth is None
            else default_max_depth
        )
        self._default_attenuation = (
            settings.risk_propagation_attenuation
            if default_attenuation is None
            else default_attenuation
        )
        self._max_affected = (
            settings.risk_propagation_max_affected
            if max_affected is None
            else max_affected
        )

    # ------------------------------------------------------------------ #
    # Main operation
    # ------------------------------------------------------------------ #

    def propagate(
        self,
        request: RiskPropagationRequest,
    ) -> RiskPropagationResponse:
        """Propagate a resolved entity's risk downstream through the graph.

        Args:
            request: Contains the resolved entity identifier, the source
                ``risk_score`` (0-100, from Module 9), and optional traversal
                parameters.

        Returns:
            A structured :class:`RiskPropagationResponse` describing the source
            entity and every downstream entity it impacts. ``propagated`` is
            ``False`` only when the entity is unresolved (no database access).

        Raises:
            ValueError: If the input is invalid (defense-in-depth; Pydantic
                already rejects invalid scores/depths at the request layer).
            EntityNotFoundError: If the source entity node does not exist in the
                graph (Module 10 never creates nodes).
            ServiceUnavailable: If Neo4j is unreachable.
            Neo4jError: If a database error occurs.
        """
        entity_id = request.entity_id
        risk_score = request.risk_score

        self._validate_risk_score(risk_score)
        max_depth = self._resolve_max_depth(request.max_depth)
        attenuation = self._resolve_attenuation(request.attenuation)
        self._validate_depth(max_depth)
        self._validate_attenuation(attenuation)

        # Unresolved entity (blank node id from Module 8): do not touch Neo4j,
        # return a controlled non-propagated response.
        if entity_id is None:
            logger.warning("Risk propagation skipped: entity is unresolved")
            return RiskPropagationResponse(
                source_entity_id="",
                source_entity_name=request.entity_name,
                source_risk_score=risk_score,
                propagated=False,
                affected_entities=[],
                affected_count=0,
                max_depth_reached=0,
                timestamp=self._utcnow(),
                error=(
                    "Entity is unresolved (no node_id provided); "
                    "no propagation was run"
                ),
            )

        # Resolve the source node (reusing the existing repository lookup).
        source_record = self._get_source_node(entity_id)
        source_name = self._extract_name(source_record, request.entity_name, entity_id)
        source_score = self._extract_score(source_record, risk_score)

        # BFS traversal downstream with depth limit and cycle prevention.
        affected, max_reached = self._traverse_downstream(
            entity_id, max_depth, request.relationship_types
        )

        affected_entities = [
            self._build_affected_entity(node_id, info, source_score, attenuation)
            for node_id, info in affected.items()
        ]
        # Keep results deterministic and bounded.
        affected_entities.sort(key=lambda e: (e.depth, e.entity_id))
        affected_entities = affected_entities[: self._max_affected]

        response = RiskPropagationResponse(
            source_entity_id=entity_id,
            source_entity_name=source_name,
            source_risk_score=source_score,
            propagated=True,
            affected_entities=affected_entities,
            affected_count=len(affected_entities),
            max_depth_reached=max_reached,
            timestamp=self._utcnow(),
        )

        logger.info(
            "Risk propagation complete",
            extra={
                "source_entity_id": entity_id,
                "affected_count": response.affected_count,
                "max_depth_reached": response.max_depth_reached,
            },
        )
        return response

    # ------------------------------------------------------------------ #
    # Graph traversal
    # ------------------------------------------------------------------ #

    def _traverse_downstream(
        self,
        source_id: str,
        max_depth: int,
        relationship_types: Optional[list[str]],
    ) -> tuple[dict[str, dict[str, Any]], int]:
        """Breadth-first traversal of downstream entities, depth-limited.

        Cycle prevention: a ``visited`` set ensures every node is recorded at
        most once (at its shallowest depth), and the loop is bounded by
        ``max_depth``, so infinite/cyclic traversal is impossible.

        Args:
            source_id: The source entity identifier.
            max_depth: Maximum number of hops to traverse.
            relationship_types: Optional relationship type filters.

        Returns:
            A 2-tuple of ``(affected, max_reached)`` where ``affected`` maps a
            downstream node id to ``{"name", "depth", "rel_type"}`` and
            ``max_reached`` is the furthest depth actually reached.

        Raises:
            ServiceUnavailable: If Neo4j is unreachable.
            Neo4jError: If a database error occurs.
        """
        visited = {source_id}
        current_level: list[str] = [source_id]
        affected: dict[str, dict[str, Any]] = {}
        max_reached = 0

        for depth in range(1, max_depth + 1):
            if not current_level:
                break
            try:
                records = self._repository.find_downstream_neighbors(
                    current_level,
                    relationship_types=relationship_types,
                    limit=self._max_affected,
                )
            except ServiceUnavailable:
                logger.error(
                    "Risk propagation failed: Neo4j unavailable",
                    extra={"source_id": source_id},
                )
                raise
            except Neo4jError as exc:
                logger.error(
                    "Risk propagation failed: Neo4j error",
                    extra={"source_id": source_id, "error": str(exc)},
                )
                raise

            next_level: list[str] = []
            for record in records:
                node = self._extract_node(record)
                node_id = node.get("id")
                if not node_id or node_id in visited:
                    continue
                visited.add(node_id)
                next_level.append(node_id)
                affected[node_id] = {
                    "name": node.get("name"),
                    "depth": depth,
                    "rel_type": self._extract_rel_type(record),
                }
            current_level = next_level
            if next_level:
                max_reached = depth

        return affected, max_reached

    def _build_affected_entity(
        self,
        node_id: str,
        info: dict[str, Any],
        source_score: float,
        attenuation: float,
    ) -> AffectedEntity:
        """Compute the propagated risk for a single affected entity.

        The propagated score is ``source_score * attenuation ** depth`` clamped
        to the 0-100 scale. The risk level is derived by delegating to the
        existing :class:`RiskService` (no duplicated risk engine).
        """
        depth = int(info["depth"])
        propagated = self._clamp(source_score * (attenuation**depth))
        propagated = round(propagated, 2)
        level = self._risk_service.calculate_risk_level(propagated)
        return AffectedEntity(
            entity_id=node_id,
            entity_name=info.get("name"),
            depth=depth,
            propagated_risk_score=propagated,
            propagated_risk_level=level,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_source_node(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Look up the source entity node, mapping not-found to a domain error."""
        try:
            record = self._repository.get_node_by_id(entity_id)
        except ServiceUnavailable:
            logger.error(
                "Risk propagation source lookup: Neo4j unavailable",
                extra={"entity_id": entity_id},
            )
            raise
        except Neo4jError as exc:
            logger.error(
                "Risk propagation source lookup: Neo4j error",
                extra={"entity_id": entity_id, "error": str(exc)},
            )
            raise

        if record is None:
            logger.warning(
                "Risk propagation: source node not found (no node created)",
                extra={"entity_id": entity_id},
            )
            raise EntityNotFoundError(
                f"Entity node with id '{entity_id}' was not found in the graph"
            )
        return record

    def _validate_risk_score(self, risk_score: float) -> None:
        """Reject risk scores outside the configured valid range."""
        if (
            isinstance(risk_score, bool)
            or not isinstance(risk_score, (int, float))
            or risk_score < self._score_min
            or risk_score > self._score_max
        ):
            raise ValueError(
                "risk_score must be between "
                f"{self._score_min:g} and {self._score_max:g} inclusive"
            )

    def _resolve_max_depth(self, value: Optional[int]) -> int:
        return self._default_max_depth if value is None else value

    def _resolve_attenuation(self, value: Optional[float]) -> float:
        return self._default_attenuation if value is None else value

    def _validate_depth(self, max_depth: int) -> None:
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")

    def _validate_attenuation(self, attenuation: float) -> None:
        if (
            isinstance(attenuation, bool)
            or not isinstance(attenuation, (int, float))
            or attenuation < 0.0
            or attenuation > 1.0
        ):
            raise ValueError("attenuation must be between 0.0 and 1.0 inclusive")

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp a value to the 0-100 risk scale."""
        if value < 0.0:
            return 0.0
        if value > 100.0:
            return 100.0
        return value

    @staticmethod
    def _extract_node(record: dict[str, Any]) -> dict[str, Any]:
        """Extract a downstream node from a traversal record.

        Handles both plain dicts (from mocks/simplified repositories) and real
        Neo4j nodes returned under the ``node``/``n`` keys.
        """
        if isinstance(record, dict):
            for key in ("node", "n"):
                value = record.get(key)
                if value is None:
                    continue
                props = dict(value.items()) if hasattr(value, "items") else dict(value)
                labels = list(value.labels) if hasattr(value, "labels") else []
                return {
                    "id": props.get("id", ""),
                    "label": labels[0] if labels else "Unknown",
                    "name": props.get("name"),
                    "properties": props,
                }
        # Fallback: the record itself is a node-shaped dict.
        if isinstance(record, dict) and "id" in record and "properties" in record:
            return {
                "id": record.get("id", ""),
                "label": record.get("label", "Unknown"),
                "name": record.get("properties", {}).get("name"),
                "properties": record.get("properties", {}),
            }
        return {"id": "", "label": "Unknown", "name": None, "properties": {}}

    @staticmethod
    def _extract_rel_type(record: dict[str, Any]) -> Optional[str]:
        """Extract the relationship type from a traversal record, if present."""
        if isinstance(record, dict):
            value = record.get("rel_type")
            if isinstance(value, str):
                return value
            rel = record.get("r")
            if rel is not None and hasattr(rel, "type"):
                return rel.type
        return None

    @staticmethod
    def _extract_name(
        node: Any, requested_name: Optional[str], entity_id: str
    ) -> Optional[str]:
        """Resolve a display name for the source entity.

        Prefers the node's ``name`` property, then the request-supplied name,
        then falls back to the node id. Handles a raw record shape
        (``{"n": <node>}``) as returned by ``get_node_by_id`` as well as a
        bare node dict.
        """
        node = _unwrap_node(node)
        name: Optional[str] = None
        if isinstance(node, dict):
            raw = node.get("name") or (node.get("properties") or {}).get("name")
            if isinstance(raw, str):
                name = raw
        elif node is not None and hasattr(node, "get"):
            raw = node.get("name")
            if isinstance(raw, str):
                name = raw
        return name or requested_name or entity_id or None

    @staticmethod
    def _extract_score(node: Any, fallback: float) -> float:
        """Return the node's persisted risk score, falling back to the request.

        The source node may already carry a ``risk_score`` set by Module 9; when
        present it is preferred, otherwise the request-supplied score is used.
        Handles a raw record shape (``{"n": <node>}``) as well as a bare node.
        """
        node = _unwrap_node(node)
        try:
            if isinstance(node, dict):
                props = node.get("properties") if isinstance(node.get("properties"), dict) else node
                raw = props.get("risk_score")
            elif node is not None and hasattr(node, "get"):
                raw = node.get("risk_score")
            else:
                raw = None
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return float(raw)
        except Exception:  # pragma: no cover - defensive
            pass
        return float(fallback)

    @staticmethod
    def _utcnow() -> datetime:
        """Return the current UTC time as a timezone-aware datetime."""
        return datetime.now(timezone.utc)

