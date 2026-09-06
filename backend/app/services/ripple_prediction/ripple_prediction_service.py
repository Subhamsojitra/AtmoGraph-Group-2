"""Ripple prediction service for AtmoGraph (Module 17, Part 3).

Orchestrates the existing risk propagation and GNN prediction services to
compute the ripple effect of a source entity's disruption through the supply
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
    """Compute the ripple effect of a source entity's risk (Module 17)."""

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
        """Compute the ripple prediction for a source entity."""
        entity_id = request.entity_id
        node = self._resolve_entity(entity_id)
        risk_score = self._resolve_risk_score(request, node)
        risk_request = RiskPropagationRequest(
            entity_id=entity_id,
            entity_name=request.entity_name,
            risk_score=risk_score,
            max_depth=request.max_depth,
            attenuation=request.attenuation,
            relationship_types=request.relationship_types,
        )
        propagation = self._run_propagation(risk_request)
        predictions = self._run_prediction(request)
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
                f"Entity '{entity_id}' not found in the graph"
            )
        return node
