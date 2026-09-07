"""Tests for Module 17 Part 3: RipplePredictionService business logic.

These tests verify the ripple prediction service in complete isolation from
Neo4j, the GNN, and the WebSocket layer. All dependencies (GraphService,
RiskPropagationService, PredictionService) are mocked, so NO database,
checkpoint, or GPU is required.

Conventions follow tests/test_risk_propagation_service.py and
tests/test_websocket_prediction.py.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import ValidationError

# Fake Neo4j env vars must be set before importing the app
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

from app.ml.exceptions import (  # noqa: E402
    GNNModelNotAvailableError,
    GNNPredictionError,
    GNNPredictionRuntimeError,
)
from app.schemas.prediction import (  # noqa: E402
    NodePrediction,
    PredictionResponse,
)
from app.schemas.risk_propagation import (  # noqa: E402
    AffectedEntity,
    RiskPropagationResponse,
)
from app.schemas.websocket import (  # noqa: E402
    WebSocketRipplePredictionRequest,
)
from app.services.prediction_service import (  # noqa: E402
    ModelNotAvailableError,
    PredictionService,
)
from app.services.ripple_prediction.exceptions import (  # noqa: E402
    RipplePredictionError,
)
from app.services.ripple_prediction.result_schema import (  # noqa: E402
    RippleAffectedEntity,
    RipplePredictionResult,
)
from app.services.ripple_prediction.ripple_prediction_service import (  # noqa: E402
    RipplePredictionService,
)
from app.services.risk.exceptions import (  # noqa: E402
    EntityNotFoundError,
)
from app.services.graph_service import GraphService  # noqa: E402
from app.services.risk_propagation.risk_propagation_service import (  # noqa: E402
    RiskPropagationService,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_affected_entity(
    entity_id: str,
    name: Optional[str] = None,
    depth: int = 1,
    risk_score: float = 50.0,
    risk_level: str = "MEDIUM",
) -> AffectedEntity:
    """Build a Module 10 AffectedEntity with sensible defaults."""
    return AffectedEntity(
        entity_id=entity_id,
        entity_name=name or f"Entity {entity_id}",
        depth=depth,
        propagated_risk_score=risk_score,
        propagated_risk_level=risk_level,
    )


def make_propagation_response(
    source_entity_id: str = "SRC_001",
    source_entity_name: str = "Source Entity",
    source_risk_score: float = 75.0,
    affected_entities: Optional[list[AffectedEntity]] = None,
    max_depth_reached: int = 2,
    propagated: bool = True,
    error: Optional[str] = None,
) -> RiskPropagationResponse:
    """Build a Module 10 RiskPropagationResponse."""
    affected = affected_entities or [
        make_affected_entity("DST_001", "Downstream A", depth=1, risk_score=50.0, risk_level="MEDIUM"),
        make_affected_entity("DST_002", "Downstream B", depth=2, risk_score=25.0, risk_level="LOW"),
    ]
    return RiskPropagationResponse(
        source_entity_id=source_entity_id,
        source_entity_name=source_entity_name,
        source_risk_score=source_risk_score,
        affected_entities=affected,
        affected_count=len(affected),
        max_depth_reached=max_depth_reached,
        propagated=propagated,
        error=error,
    )


def make_prediction_response(
    *pairs: tuple[str, float],
) -> PredictionResponse:
    """Build a Module 14 PredictionResponse from (node_id, value) pairs."""
    predictions = [
        NodePrediction(node_id=node_id, prediction=value)
        for node_id, value in pairs
    ]
    return PredictionResponse(
        predictions=predictions, prediction_count=len(predictions)
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_graph_service() -> MagicMock:
    """Mock GraphService for entity resolution."""
    return MagicMock(spec=GraphService)


@pytest.fixture
def mock_risk_propagation_service() -> MagicMock:
    """Mock RiskPropagationService for risk propagation."""
    return MagicMock(spec=RiskPropagationService)


@pytest.fixture
def mock_prediction_service() -> MagicMock:
    """Mock PredictionService for GNN inference."""
    return MagicMock(spec=PredictionService)


@pytest.fixture
def service(
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
) -> RipplePredictionService:
    """RipplePredictionService with all dependencies mocked."""
    return RipplePredictionService(
        graph_service=mock_graph_service,
        risk_propagation_service=mock_risk_propagation_service,
        prediction_service=mock_prediction_service,
    )


@pytest.fixture
def base_request() -> WebSocketRipplePredictionRequest:
    """A valid WebSocketRipplePredictionRequest for testing."""
    return WebSocketRipplePredictionRequest(
        entity_id="SRC_001",
        entity_name="Source Entity",
        risk_score=75.0,
        max_depth=3,
        attenuation=0.8,
    )


@pytest.fixture
def mock_node() -> dict[str, Any]:
    """A mock graph node response."""
    return {
        "id": "SRC_001",
        "properties": {"name": "Source Entity", "risk_score": 60.0},
    }
# --------------------------------------------------------------------------- #
# 1. Successful ripple prediction
# --------------------------------------------------------------------------- #


def test_successful_ripple_prediction(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """Successful ripple prediction returns structured result with real data."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = make_propagation_response()
    mock_prediction_service.get_predictions.return_value = make_prediction_response(
        ("DST_001", 0.75), ("DST_002", 0.25)
    )

    result = service.predict(base_request)

    assert isinstance(result, RipplePredictionResult)
    assert result.source_entity_id == "SRC_001"
    assert result.source_entity_name == "Source Entity"
    assert result.source_risk_score == 75.0
    assert result.propagated is True
    assert result.affected_count == 2
    assert result.prediction_count == 2
    assert result.error is None
    assert len(result.affected_entities) == 2

    # Verify affected entities have real GNN predictions
    pred_map = {ae.entity_id: ae for ae in result.affected_entities}
    assert pred_map["DST_001"].gnn_prediction == 0.75
    assert pred_map["DST_002"].gnn_prediction == 0.25
    assert pred_map["DST_001"].propagated_risk_score == 50.0
    assert pred_map["DST_001"].propagated_risk_level == "MEDIUM"

    # Verify correct interaction with dependencies
    mock_graph_service.get_node_by_id.assert_called_once_with("SRC_001")
    mock_risk_propagation_service.propagate.assert_called_once()
    mock_prediction_service.get_predictions.assert_called_once()


def test_ripple_prediction_enriches_affected_entities_with_gnn(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """Affected entities are enriched with matching GNN predictions by node_id."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = make_propagation_response(
        affected_entities=[
            make_affected_entity("DST_001", depth=1, risk_score=50.0),
            make_affected_entity("DST_002", depth=2, risk_score=25.0),
            make_affected_entity("DST_003", depth=2, risk_score=10.0),
        ],
    )
    # Only DST_001 and DST_003 have predictions; DST_002 has None
    mock_prediction_service.get_predictions.return_value = make_prediction_response(
        ("DST_001", 0.8), ("DST_003", 0.3)
    )

    result = service.predict(base_request)

    assert result.affected_count == 3
    assert result.prediction_count == 2
    pred_map = {ae.entity_id: ae for ae in result.affected_entities}
    assert pred_map["DST_001"].gnn_prediction == 0.8
    assert pred_map["DST_002"].gnn_prediction is None  # No prediction available
    assert pred_map["DST_003"].gnn_prediction == 0.3


def test_ripple_prediction_no_fabricated_entities(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """Affected entities come ONLY from the real propagation output."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = make_propagation_response(
        affected_entities=[
            make_affected_entity("REAL_001", depth=1, risk_score=50.0),
        ],
    )
    mock_prediction_service.get_predictions.return_value = make_prediction_response(
        ("REAL_001", 0.9), ("FAKE_001", 0.1)  # FAKE_001 is NOT in propagation
    )

    result = service.predict(base_request)

    # Only REAL_001 should appear; FAKE_001 must NOT be fabricated
    assert result.affected_count == 1
    assert result.affected_entities[0].entity_id == "REAL_001"
    entity_ids = [ae.entity_id for ae in result.affected_entities]
    assert "FAKE_001" not in entity_ids


def test_ripple_prediction_no_fabricated_predictions(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """GNN predictions are real; missing predictions are None, not invented."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = make_propagation_response(
        affected_entities=[
            make_affected_entity("DST_001", depth=1, risk_score=50.0),
        ],
    )
    # No predictions returned at all
    mock_prediction_service.get_predictions.return_value = make_prediction_response()

    result = service.predict(base_request)

    assert result.prediction_count == 0
    assert result.affected_entities[0].gnn_prediction is None
# --------------------------------------------------------------------------- #
# 2. Valid source entity
# --------------------------------------------------------------------------- #


def test_valid_source_entity_resolved(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """A valid source entity is resolved before propagation."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = make_propagation_response()
    mock_prediction_service.get_predictions.return_value = make_prediction_response()

    result = service.predict(base_request)

    assert result.propagated is True
    mock_graph_service.get_node_by_id.assert_called_once_with("SRC_001")


def test_entity_id_trimmed_in_request() -> None:
    """entity_id is trimmed by the schema validator."""
    req = WebSocketRipplePredictionRequest(entity_id="  SRC_001  ")
    assert req.entity_id == "SRC_001"


def test_entity_name_normalized_to_none_when_blank() -> None:
    """Blank entity_name becomes None."""
    req = WebSocketRipplePredictionRequest(entity_id="SRC_001", entity_name="   ")
    assert req.entity_name is None
# --------------------------------------------------------------------------- #
# 3. Unknown source entity
# --------------------------------------------------------------------------- #


def test_unknown_source_entity_raises_entity_not_found(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
) -> None:
    """Unknown source entity raises EntityNotFoundError, not empty result."""
    mock_graph_service.get_node_by_id.return_value = None

    with pytest.raises(EntityNotFoundError, match="not found"):
        service.predict(base_request)

    # Propagation and prediction must NOT run for unknown entity
    mock_risk_propagation_service.propagate.assert_not_called()
    mock_prediction_service.get_predictions.assert_not_called()


def test_unknown_entity_fails_fast_before_propagation(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
) -> None:
    """Entity resolution happens BEFORE propagation."""
    mock_graph_service.get_node_by_id.return_value = None

    request = WebSocketRipplePredictionRequest(entity_id="NONEXISTENT")

    with pytest.raises(EntityNotFoundError):
        service.predict(request)

    mock_risk_propagation_service.propagate.assert_not_called()
# --------------------------------------------------------------------------- #
# 4. Missing/invalid entity
# --------------------------------------------------------------------------- #


def test_missing_entity_id_raises_validation_error() -> None:
    """Missing entity_id fails at schema validation."""
    with pytest.raises(ValidationError):
        WebSocketRipplePredictionRequest()  # type: ignore[call-arg]


def test_blank_entity_id_raises_validation_error() -> None:
    """Blank entity_id fails at schema validation."""
    with pytest.raises(ValidationError, match="blank"):
        WebSocketRipplePredictionRequest(entity_id="   ")


def test_null_entity_id_raises_validation_error() -> None:
    """Null entity_id fails at schema validation."""
    with pytest.raises(ValidationError):
        WebSocketRipplePredictionRequest(entity_id=None)  # type: ignore[arg-type]


def test_invalid_risk_score_rejected() -> None:
    """Risk score outside [0, 100] is rejected."""
    with pytest.raises(ValidationError):
        WebSocketRipplePredictionRequest(entity_id="SRC_001", risk_score=150.0)


def test_invalid_max_depth_rejected() -> None:
    """max_depth outside [1, 20] is rejected."""
    with pytest.raises(ValidationError):
        WebSocketRipplePredictionRequest(entity_id="SRC_001", max_depth=25)


def test_invalid_attenuation_rejected() -> None:
    """attenuation outside [0, 1] is rejected."""
    with pytest.raises(ValidationError):
        WebSocketRipplePredictionRequest(entity_id="SRC_001", attenuation=1.5)
# --------------------------------------------------------------------------- #
# 5. Neo4j unavailable
# --------------------------------------------------------------------------- #


def test_neo4j_unavailable_during_entity_resolution(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
) -> None:
    """Neo4j unavailable during entity resolution raises ServiceUnavailable."""
    mock_graph_service.get_node_by_id.side_effect = ServiceUnavailable("down")

    with pytest.raises(ServiceUnavailable):
        service.predict(base_request)

    mock_risk_propagation_service.propagate.assert_not_called()
    mock_prediction_service.get_predictions.assert_not_called()


def test_neo4j_unavailable_during_propagation(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """Neo4j unavailable during propagation raises ServiceUnavailable."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.side_effect = ServiceUnavailable("down")

    with pytest.raises(ServiceUnavailable):
        service.predict(base_request)

    mock_prediction_service.get_predictions.assert_not_called()


def test_neo4j_error_during_entity_resolution(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
) -> None:
    """Neo4jError during entity resolution propagates."""
    mock_graph_service.get_node_by_id.side_effect = Neo4jError("db error")

    with pytest.raises(Neo4jError):
        service.predict(base_request)
# --------------------------------------------------------------------------- #
# 6. Empty graph
# --------------------------------------------------------------------------- #


def test_empty_graph_propagation_returns_no_affected_entities(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """Empty graph propagation returns zero affected entities."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = make_propagation_response(
        affected_entities=[],
        affected_count=0,
        max_depth_reached=0,
    )
    mock_prediction_service.get_predictions.return_value = make_prediction_response()

    result = service.predict(base_request)

    assert result.affected_count == 0
    assert result.affected_entities == []
    assert result.propagated is True
    assert result.prediction_count == 0
# --------------------------------------------------------------------------- #
# 7. Risk propagation failure
# --------------------------------------------------------------------------- #


def test_risk_propagation_failure_raises_ripple_error(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """Unexpected risk propagation failure raises RipplePredictionError."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.side_effect = RuntimeError("propagation boom")

    with pytest.raises(RipplePredictionError, match="unexpected error"):
        service.predict(base_request)

    mock_prediction_service.get_predictions.assert_not_called()


def test_risk_propagation_entity_not_found_propagates(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """EntityNotFoundError from propagation propagates."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.side_effect = EntityNotFoundError("gone")

    with pytest.raises(EntityNotFoundError):
        service.predict(base_request)


def test_risk_propagation_unexpected_type_raises_ripple_error(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """Propagation returning wrong type raises RipplePredictionError."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = "not a response"

    with pytest.raises(RipplePredictionError, match="unexpected result type"):
        service.predict(base_request)
# --------------------------------------------------------------------------- #
# 8. GNN prediction failure
# --------------------------------------------------------------------------- #


def test_gnn_prediction_failure_raises_gnn_error(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """GNN prediction failure raises GNNPredictionError."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = make_propagation_response()
    mock_prediction_service.get_predictions.side_effect = GNNPredictionRuntimeError("boom")

    with pytest.raises(GNNPredictionRuntimeError):
        service.predict(base_request)


def test_gnn_unexpected_error_raises_ripple_error(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """Unexpected GNN prediction error raises RipplePredictionError."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = make_propagation_response()
    mock_prediction_service.get_predictions.side_effect = RuntimeError("gnn boom")

    with pytest.raises(RipplePredictionError, match="unexpected error"):
        service.predict(base_request)
# --------------------------------------------------------------------------- #
# 9. Missing checkpoint
# --------------------------------------------------------------------------- #


def test_missing_checkpoint_returns_result_without_predictions(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """Missing GNN checkpoint returns result with prediction_count=0."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = make_propagation_response()
    mock_prediction_service.get_predictions.side_effect = ModelNotAvailableError("no checkpoint")

    result = service.predict(base_request)

    # Result still has real propagation data
    assert result.propagated is True
    assert result.affected_count == 2
    assert result.prediction_count == 0
    assert all(ae.gnn_prediction is None for ae in result.affected_entities)
# --------------------------------------------------------------------------- #
# 10. Result schema serialization
# --------------------------------------------------------------------------- #


def test_result_schema_is_json_serializable(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """RipplePredictionResult is JSON-serializable."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = make_propagation_response()
    mock_prediction_service.get_predictions.return_value = make_prediction_response(
        ("DST_001", 0.75)
    )

    result = service.predict(base_request)
    json_data = result.model_dump(mode="json")

    assert isinstance(json_data, dict)
    assert json_data["source_entity_id"] == "SRC_001"
    assert json_data["affected_count"] == 2
    assert json_data["prediction_count"] == 1
    assert "timestamp" in json_data
    assert isinstance(json_data["timestamp"], str)


def test_result_schema_model_dump_dict(
    service: RipplePredictionService,
    mock_graph_service: MagicMock,
    mock_risk_propagation_service: MagicMock,
    mock_prediction_service: MagicMock,
    base_request: WebSocketRipplePredictionRequest,
    mock_node: dict[str, Any],
) -> None:
    """RipplePredictionResult.model_dump() returns a dict."""
    mock_graph_service.get_node_by_id.return_value = mock_node
    mock_risk_propagation_service.propagate.return_value = make_propagation_response()
    mock_prediction_service.get_predictions.return_value = make_prediction_response()

    result = service.predict(base_request)
    data = result.model_dump()

    assert isinstance(data, dict)
    assert data["source_entity_id"] == "SRC_001"
    assert data["propagated"] is True


def test_ripple_affected_entity_serialization() -> None:
    """RippleAffectedEntity is JSON-serializable."""
    entity = RippleAffectedEntity(
        entity_id="DST_001",
        entity_name="Test Entity",
        depth=1,
        propagated_risk_score=50.0,
        propagated_risk_level="MEDIUM",
        gnn_prediction=0.75,
    )
    data = entity.model_dump(mode="json")
    assert data["entity_id"] == "DST_001"
    assert data["gnn_prediction"] == 0.75


def test_ripple_affected_entity_with_none_prediction() -> None:
    """RippleAffectedEntity with None prediction serializes correctly."""
    entity = RippleAffectedEntity(
        entity_id="DST_001",
        entity_name=None,
        depth=1,
        propagated_risk_score=50.0,
        propagated_risk_level="MEDIUM",
        gnn_prediction=None,
    )
    data = entity.model_dump(mode="json")
    assert data["gnn_prediction"] is None
    assert data["entity_name"] is None
