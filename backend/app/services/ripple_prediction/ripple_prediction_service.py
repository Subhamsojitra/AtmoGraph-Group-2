"""Ripple prediction service for AtmoGraph (Module 17, Part 3).

Orchestrates the existing risk propagation and GNN prediction services to
compute the ripple effect of a source entity\'s disruption through the supply
chain. This is a TRANSPORT-AGNOSTIC business layer: it returns a structured
Python result and knows nothing about WebSocket connections.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.logger import get_logger
from app.ml.exceptions import GNNPredictionError
from app.schemas.prediction import PredictionRequest
from app.schemas.risk_propagation import (
    RiskPropagationRequest,
    RiskPropagationResponse,
)
from app.schemas.websocket import WebSocketRipplePredictionRequest
from app.services.graph_service import GraphService
from app.services.prediction_service import ModelNotAvailableError, PredictionService
from app.services.risk.exceptions import EntityNotFoundError
from app.services.ripple_prediction.exceptions import RipplePredictionError
from app.services.ripple_prediction.result_schema import (
    RippleAffectedEntity,
    RipplePredictionResult,
)
from app.services.risk_propagation.risk_propagation_service import (
    RiskPropagationService,
)

logger = get_logger(__name__)

__all__ = [
    "RipplePredictionError",
    "RipplePredictionService",
]


class RipplePredictionService:
    """Compute the ripple effect of a source entity\'s risk (Module 17).

    Accepts a validated WebSocketRipplePredictionRequest, resolves the source
    entity through the existing graph layer, propagates its risk downstream
    using the existing Module 10 engine, and enriches the affected entities
    with real GNN predictions from the existing Module 14 pipeline.

    External callers (the future WebSocket layer) use this service. It never
    touches a WebSocket and never invents data.
    """

    def __init__(
        self,
        graph_service: Optional[GraphService] = None,
        risk_propagation_service: Optional[RiskPropagationService] = None,
        prediction_service: Optional[PredictionService] = None,
    ) -> None:
        self._graph_service = graph_service or GraphService()
        self._risk_propagation_service = (
            risk_propagation_service or RiskPropagationService()
        )
        self._prediction_service = prediction_service or PredictionService()

    def predict(
        self, request: WebSocketRipplePredictionRequest
    ) -> RipplePredictionResult:
        """Compute the ripple prediction for a source entity.

        Args:
            request: A validated WebSocketRipplePredictionRequest carrying at
                least the source entity_id.

        Returns:
            A structured RipplePredictionResult containing the affected
            entities (from the real Module 10 propagation) enriched with
            real GNN predictions (from the real Module 14 pipeline).

        Raises:
            EntityNotFoundError: The source entity does not exist.
            ServiceUnavailable: Neo4j is unreachable.
            ModelNotAvailableError: No trained GNN checkpoint configured.
            GNNPredictionError: GNN inference fails.
            RipplePredictionError: Unexpected internal failure.
        """
        entity_id = request.entity_id

        # 1. Resolve the source entity BEFORE propagation so an unknown
        #    entity fails fast with a precise, typed error.
        node = self._resolve_entity(entity_id)

        # 2. Determine the risk score: request-supplied wins; when omitted,
        #    the entity\'s persisted risk score from Module 9 is used.
        risk_score = self._resolve_risk_score(request, node)

        # 3. Build a Module 10 risk propagation request and run the existing
        #    propagation engine.
        risk_request = RiskPropagationRequest(
            entity_id=entity_id,
            entity_name=request.entity_name,
            risk_score=risk_score,
            max_depth=request.max_depth,
            attenuation=request.attenuation,
            relationship_types=request.relationship_types,
        )
        propagation = self._run_propagation(risk_request)

        # 4. Run the existing Module 14 GNN prediction for the current graph.
        predictions = self._run_prediction(request)

        # 5. Enrich affected entities with real GNN predictions and build the
        #    structured result.
        result = self._build_result(propagation, predictions)

        logger.info(
            "Ripple prediction computed",
            extra={
                "source_entity_id": result.source_entity_id,
                "affected_count": result.affected_count,
                "prediction_count": result.prediction_count,
                "propagated": result.propagated,
            },
        )
        return result

    def _resolve_entity(self, entity_id: str) -> object:
        """Resolve the source entity node, raising if it does not exist."""
        try:
            node = self._graph_service.get_node_by_id(entity_id)
        except ServiceUnavailable:
            logger.error(
                "Ripple prediction: Neo4j unavailable during entity resolution",
                extra={"entity_id": entity_id},
            )
            raise
        except Neo4jError as exc:
            logger.error(
                "Ripple prediction: Neo4j error during entity resolution",
                extra={"entity_id": entity_id, "error": str(exc)},
            )
            raise
        except Exception as exc:
            logger.error(
                "Ripple prediction: unexpected error during entity resolution",
                extra={"entity_id": entity_id, "error": str(exc)},
                exc_info=True,
            )
            raise RipplePredictionError(
                "Entity resolution failed due to an unexpected error"
            ) from exc

        if node is None:
            logger.info(
                "Ripple prediction: source entity not found",
                extra={"entity_id": entity_id},
            )
            raise EntityNotFoundError(
                f"Entity \'{entity_id}\' not found in the graph"
            )
        return node

    def _resolve_risk_score(
        self,
        request: WebSocketRipplePredictionRequest,
        node: object,
    ) -> float:
        """Determine the source risk score.

        Prefers the request-supplied score. When omitted, falls back to the
        entity\'s persisted risk_score property. If neither is available,
        defaults to 0.0 (LOW) \u2014 never invents a non-zero risk.
        """
        if request.risk_score is not None:
            return float(request.risk_score)

        persisted = self._extract_persisted_risk_score(node)
        if persisted is not None:
            return persisted

        logger.info(
            "Ripple prediction: no risk score provided or persisted; "
            "defaulting to 0.0",
            extra={"entity_id": request.entity_id},
        )
        return 0.0

    @staticmethod
    def _extract_persisted_risk_score(node: object) -> Optional[float]:
        """Extract the persisted risk_score from a graph node response."""
        properties: Optional[dict] = None

        if hasattr(node, "properties"):
            candidate = getattr(node, "properties", None)
            if isinstance(candidate, dict):
                properties = candidate
        elif isinstance(node, dict):
            candidate = node.get("properties")
            if isinstance(candidate, dict):
                properties = candidate
            elif "risk_score" in node:
                properties = node

        if not properties:
            return None

        raw = properties.get("risk_score")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        return None

    def _run_propagation(
        self, risk_request: RiskPropagationRequest
    ) -> RiskPropagationResponse:
        """Run the existing Module 10 risk propagation engine."""
        try:
            response = self._risk_propagation_service.propagate(risk_request)
        except EntityNotFoundError:
            raise
        except ServiceUnavailable:
            logger.error("Ripple prediction: Neo4j unavailable during propagation")
            raise
        except Neo4jError as exc:
            logger.error(
                "Ripple prediction: Neo4j error during propagation",
                extra={"error": str(exc)},
            )
            raise
        except Exception as exc:
            logger.error(
                "Ripple prediction: unexpected error during propagation",
                extra={"error": str(exc)},
                exc_info=True,
            )
            raise RipplePredictionError(
                "Risk propagation failed due to an unexpected error"
            ) from exc

        if not isinstance(response, RiskPropagationResponse):
            raise RipplePredictionError(
                "Risk propagation returned an unexpected result type"
            )
        return response

    def _run_prediction(
        self, request: WebSocketRipplePredictionRequest
    ) -> Optional[list]:
        """Run the existing Module 14 GNN prediction for the current graph.

        Returns a list of NodePrediction entries, or None when prediction is
        unavailable (e.g. no checkpoint configured). Returns None instead of
        raising when the model is simply not deployed.
        """
        prediction_request = PredictionRequest(
            relationship_types=request.relationship_types,
        )
        try:
            response = self._prediction_service.get_predictions(prediction_request)
        except ModelNotAvailableError:
            logger.warning(
                "Ripple prediction: no GNN checkpoint configured; "
                "returning result without predictions",
            )
            return None
        except ServiceUnavailable:
            logger.error(
                "Ripple prediction: Neo4j unavailable during GNN prediction"
            )
            raise
        except GNNPredictionError as exc:
            logger.error(
                "Ripple prediction: GNN inference failed",
                extra={"error": str(exc)},
            )
            raise
        except Exception as exc:
            logger.error(
                "Ripple prediction: unexpected error during GNN prediction",
                extra={"error": str(exc)},
                exc_info=True,
            )
            raise RipplePredictionError(
                "GNN prediction failed due to an unexpected error"
            ) from exc

        return list(response.predictions) if response.predictions else []

    def _build_result(
        self,
        propagation: RiskPropagationResponse,
        predictions: Optional[list],
    ) -> RipplePredictionResult:
        """Build the final structured result from propagation + predictions.

        Enriches each affected entity with the matching GNN prediction (by
        node_id). Entities without a matching prediction get
        ``gnn_prediction=None`` \u2014 nothing is invented.
        """
        prediction_map: dict[str, float] = {}
        if predictions:
            for pred in predictions:
                prediction_map[pred.node_id] = pred.prediction

        affected_entities = [
            RippleAffectedEntity(
                entity_id=aff.entity_id,
                entity_name=aff.entity_name,
                depth=aff.depth,
                propagated_risk_score=aff.propagated_risk_score,
                propagated_risk_level=aff.propagated_risk_level,
                gnn_prediction=prediction_map.get(aff.entity_id),
            )
            for aff in propagation.affected_entities
        ]

        prediction_count = sum(
            1 for ae in affected_entities if ae.gnn_prediction is not None
        )

        return RipplePredictionResult(
            source_entity_id=propagation.source_entity_id,
            source_entity_name=propagation.source_entity_name,
            source_risk_score=propagation.source_risk_score,
            propagated=propagation.propagated,
            affected_entities=affected_entities,
            affected_count=propagation.affected_count,
            max_depth_reached=propagation.max_depth_reached,
            prediction_count=prediction_count,
            timestamp=datetime.now(timezone.utc),
            error=propagation.error,
        )
