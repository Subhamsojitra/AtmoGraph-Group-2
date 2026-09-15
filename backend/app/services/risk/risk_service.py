"""Risk service layer for AtmoGraph (Module 9).

This module provides the business-level risk update operations consumed by the
API layer. It depends on the existing :class:`GraphRepository` for raw database
access and does not create or manage Neo4j drivers directly.

Architecture:

    FastAPI
        ↓
    Risk API (``/api/v1/news/risk-update``)
        ↓
    RiskService (this module)
        ↓
    GraphRepository
        ↓
    Neo4jDatabase
        ↓
    Neo4j

Behaviour summary
-----------------
* Module 9 updates the risk state of an entity that has *already* been resolved
  by Module 8. It never performs entity recognition or entity resolution again
  and it never creates new graph nodes (the update query only ``MATCH``es an
  existing node).
* Risk scores are on a 0-100 scale. Out-of-range values are rejected with a
  ``ValueError`` (they are never silently clamped).
* Risk levels are derived from the score using configurable 0-100 thresholds.
  The default thresholds (LOW 0-30 / MEDIUM 31-70 / HIGH 71-90 / CRITICAL
  91-100) are an explicitly-documented **assumption** and are tunable via the
  ``RISK_LEVEL_*`` settings. They are not part of an official project spec yet,
  so they are implemented as configuration rather than pretended to be the
  official definition.
* The update is **atomic** and **parameterized**: a single transactional query
  reads the previous risk state and writes the new one inside one Neo4j write
  transaction, which avoids read-then-write race conditions.
* All database queries are parameterized (no string interpolation of user
  input), protecting against Cypher injection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.config import settings
from app.core.logger import get_logger
from app.repositories.graph_repository import GraphRepository
from app.schemas.risk import RiskLevelUpdateRequest, RiskStateUpdateResponse
from app.services.risk.exceptions import EntityNotFoundError

logger = get_logger(__name__)


# Single parameterized, atomic update. ``MATCH`` (not ``CREATE``/``MERGE``)
# guarantees Module 9 never creates a new node. The ``WITH`` captures the
# previous properties before ``SET`` overwrites them, so the response can report
# the old -> new risk transition without a separate read query.
_RISK_UPDATE_QUERY = """
MATCH (n)
WHERE n.id = $entity_id
WITH n, n.risk_score AS prev_score, n.risk_level AS prev_level
SET n.risk_score = $risk_score,
    n.risk_level = $risk_level
RETURN n, prev_score, prev_level
"""


class RiskService:
    """Application-level risk state update operations (Module 9).

    External callers (the API layer) use this service instead of accessing the
    repository or database manager directly. Business logic lives here rather
    than inside FastAPI route handlers.
    """

    def __init__(
        self,
        repository: Optional[GraphRepository] = None,
        score_min: Optional[float] = None,
        score_max: Optional[float] = None,
        medium_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
    ) -> None:
        """Initialize a risk service.

        Args:
            repository: Existing :class:`GraphRepository` to reuse. Defaults to
                a new instance bound to the existing Neo4j connection.
            score_min: Lowest accepted risk score (default from settings).
            score_max: Highest accepted risk score (default from settings).
            medium_threshold: Score at/below which the level is LOW/MEDIUM.
            high_threshold: Score at which the level becomes HIGH.
            critical_threshold: Score at which the level becomes CRITICAL.
        """
        self._repository = repository or GraphRepository()
        self._score_min = settings.risk_score_min if score_min is None else score_min
        self._score_max = settings.risk_score_max if score_max is None else score_max
        self._medium_threshold = (
            settings.risk_level_medium if medium_threshold is None else medium_threshold
        )
        self._high_threshold = settings.risk_level_high if high_threshold is None else high_threshold
        self._critical_threshold = (
            settings.risk_level_critical if critical_threshold is None else critical_threshold
        )

    # ------------------------------------------------------------------ #
    # Risk level calculation
    # ------------------------------------------------------------------ #

    def calculate_risk_level(self, risk_score: float) -> str:
        """Map a 0-100 risk score to a LOW/MEDIUM/HIGH/CRITICAL level.

        Args:
            risk_score: A score on the 0-100 scale.

        Returns:
            One of ``LOW``, ``MEDIUM``, ``HIGH``, or ``CRITICAL``.

        Raises:
            ValueError: If the score is outside the configured valid range.
        """
        self._validate_risk_score(risk_score)
        if risk_score >= self._critical_threshold:
            return "CRITICAL"
        if risk_score >= self._high_threshold:
            return "HIGH"
        if risk_score >= self._medium_threshold:
            return "MEDIUM"
        return "LOW"

    def _validate_risk_score(self, risk_score: float) -> None:
        """Reject risk scores outside the configured valid range.

        Args:
            risk_score: Candidate score to validate.

        Raises:
            ValueError: If the score is not a finite number within range.
        """
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

    # ------------------------------------------------------------------ #
    # Main operation
    # ------------------------------------------------------------------ #

    def update_risk_state(
        self,
        update_request: RiskLevelUpdateRequest,
    ) -> RiskStateUpdateResponse:
        """Update the risk state of an already-resolved entity.

        Args:
            update_request: Contains the resolved entity identifier, the new
                ``risk_score`` (0-100), and an optional reason.

        Returns:
            A structured :class:`RiskStateUpdateResponse` describing the result.
            ``updated`` is ``False`` when the entity is unresolved (blank node
            id) and the service did not write to the graph.

        Raises:
            ValueError: If the risk score is outside the valid range.
            EntityNotFoundError: If the entity id is provided but no matching
                graph node exists (Module 9 never creates nodes).
            ServiceUnavailable: If Neo4j is unavailable.
            Neo4jError: If the database write fails.
        """
        entity_id = self._normalize_entity_id(update_request.entity_id)
        risk_score = update_request.risk_score
        reason = update_request.reason

        logger.info(
            "Risk update requested",
            extra={"entity_id": entity_id, "new_risk_score": risk_score, "reason": reason},
        )

        # Defense-in-depth score validation (Pydantic already rejects these at
        # the request layer; this also protects direct service callers).
        self._validate_risk_score(risk_score)
        new_risk_level = self.calculate_risk_level(risk_score)

        # Unresolved entity (blank node id from Module 8): do not touch Neo4j,
        # return a controlled non-success response.
        if entity_id is None:
            logger.warning("Risk update skipped: entity is unresolved")
            return RiskStateUpdateResponse(
                entity_id="",
                entity_name=update_request.entity_name,
                updated=False,
                new_risk_score=risk_score,
                new_risk_level=new_risk_level,
                reason=reason,
                timestamp=self._utcnow(),
                error="Entity is unresolved (no node_id provided); no graph node was updated",
            )

        parameters = {
            "entity_id": entity_id,
            "risk_score": risk_score,
            "risk_level": new_risk_level,
        }

        try:
            records = self._repository.execute_write(_RISK_UPDATE_QUERY, parameters)
        except ServiceUnavailable:
            logger.error(
                "Risk update failed: Neo4j unavailable",
                extra={"entity_id": entity_id},
            )
            raise
        except Neo4jError as exc:
            logger.error(
                "Risk update failed: Neo4j error",
                extra={"entity_id": entity_id, "error": str(exc)},
            )
            raise

        if not records:
            logger.warning(
                "Risk update: node not found (no node created)",
                extra={"entity_id": entity_id},
            )
            raise EntityNotFoundError(
                f"Entity node with id '{entity_id}' was not found in the graph"
            )

        record = records[0] if isinstance(records, list) else records
        if isinstance(record, dict):
            node = record.get("n")
            previous_score = record.get("prev_score")
            previous_level = record.get("prev_level")
        else:
            node = getattr(record, "n", None)
            previous_score = getattr(record, "prev_score", None)
            previous_level = getattr(record, "prev_level", None)

        entity_name = self._extract_name(node, update_request.entity_name, entity_id)

        response = RiskStateUpdateResponse(
            entity_id=entity_id,
            entity_name=entity_name,
            updated=True,
            previous_risk_score=self._as_float(previous_score),
            new_risk_score=float(risk_score),
            previous_risk_level=self._as_str(previous_level),
            new_risk_level=new_risk_level,
            reason=reason,
            timestamp=self._utcnow(),
        )

        logger.info(
            "Risk update successful",
            extra={
                "entity_id": entity_id,
                "previous_risk_score": response.previous_risk_score,
                "new_risk_score": response.new_risk_score,
                "previous_risk_level": response.previous_risk_level,
                "new_risk_level": response.new_risk_level,
            },
        )
        return response

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_entity_id(entity_id: Optional[str]) -> Optional[str]:
        """Normalize/validate an entity id.

        ``None`` and whitespace-only values resolve to ``None`` (unresolved).
        Non-string values are rejected.

        Args:
            entity_id: Raw node identifier from the request.

        Returns:
            Trimmed identifier, or ``None`` when unavailable/blank.

        Raises:
            ValueError: If ``entity_id`` is not a string when provided.
        """
        if entity_id is None:
            return None
        if not isinstance(entity_id, str):
            raise ValueError("entity_id must be a string")
        stripped = entity_id.strip()
        return stripped or None

    @staticmethod
    def _extract_name(node: Any, requested_name: Optional[str], entity_id: str) -> Optional[str]:
        """Resolve a display name for the response.

        Prefers the node's ``name`` property, then the request-supplied name,
        then falls back to the node id.

        Args:
            node: A Neo4j node object or plain property dict, or ``None``.
            requested_name: Optional ``entity_name`` from the request.
            entity_id: The resolved node identifier (fallback name).

        Returns:
            A display name string, or ``None`` if nothing is available.
        """
        name: Optional[str] = None
        if isinstance(node, dict):
            raw = node.get("name")
            if isinstance(raw, str):
                name = raw
        elif node is not None and hasattr(node, "get"):
            raw = node.get("name")
            if isinstance(raw, str):
                name = raw
        return name or requested_name or entity_id or None

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        """Return the value as float when numeric, otherwise None."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    @staticmethod
    def _as_str(value: Any) -> Optional[str]:
        """Return the value as a string only when it already is one."""
        return value if isinstance(value, str) else None

    @staticmethod
    def _utcnow() -> datetime:
        """Return the current UTC time as a timezone-aware datetime."""
        return datetime.now(timezone.utc)