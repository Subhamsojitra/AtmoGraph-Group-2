"""Unit tests for the risk propagation service layer (Module 10).

These tests mock the :class:`GraphRepository` and :class:`RiskService` so they
do NOT require a running Neo4j server. Every test exercises the
:class:`RiskPropagationService` business logic in isolation.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import pytest
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import ValidationError

from app.repositories.graph_repository import GraphRepository
from app.schemas.risk_propagation import RiskPropagationRequest
from app.services.risk.exceptions import EntityNotFoundError
from app.services.risk.risk_service import RiskService
from app.services.risk_propagation.risk_propagation_service import RiskPropagationService


def _source_record(
    node_id: str = "entity-001",
    name: str = "Port of Rotterdam",
    risk_score: float | None = None,
) -> dict[str, object]:
    """A source-node record as returned by ``get_node_by_id``."""
    node: dict[str, object] = {"id": node_id, "name": name}
    if risk_score is not None:
        node["risk_score"] = risk_score
    return {"n": node}


def _neighbor_record(
    node_id: str,
    name: str,
    rel_type: str = "SUPPLIES",
) -> dict[str, object]:
    """A downstream-neighbor record as returned by ``find_downstream_neighbors``."""
    return {
        "node": {"id": node_id, "name": name},
        "rel_type": rel_type,
        "source_id": "entity-001",
    }


def _request(
    risk_score: float = 80.0,
    entity_id: str = "entity-001",
    entity_name: str = "Port of Rotterdam",
    max_depth: int | None = 2,
    attenuation: float = 0.5,
    relationship_types: list[str] | None = None,
) -> RiskPropagationRequest:
    return RiskPropagationRequest(
        entity_id=entity_id,
        entity_name=entity_name,
        risk_score=risk_score,
        max_depth=max_depth,
        attenuation=attenuation,
        relationship_types=relationship_types,
    )


@pytest.fixture()
def mock_repository() -> Mock:
    """A GraphRepository mock used to isolate the service from Neo4j."""
    return Mock(spec=GraphRepository)


@pytest.fixture()
def mock_risk_service() -> Mock:
    """A RiskService mock used to isolate the risk-level engine."""
    risk = Mock(spec=RiskService)

    def _level(score: float) -> str:
        if score >= 91:
            return "CRITICAL"
        if score >= 71:
            return "HIGH"
        if score >= 31:
            return "MEDIUM"
        return "LOW"

    risk.calculate_risk_level.side_effect = _level
    return risk


@pytest.fixture()
def service(
    mock_repository: Mock, mock_risk_service: Mock
) -> RiskPropagationService:
    return RiskPropagationService(
        repository=mock_repository,
        risk_service=mock_risk_service,
    )


# --------------------------------------------------------------------------- #
# 1. Successful propagation
# --------------------------------------------------------------------------- #


def test_successful_propagation(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record(risk_score=80.0)
    mock_repository.find_downstream_neighbors.side_effect = [
        [_neighbor_record("entity-002", "Gigafactory Assembly")],
        [_neighbor_record("entity-003", "Regional Warehouse Hub")],
    ]

    response = service.propagate(_request(risk_score=80.0))

    assert response.propagated is True
    assert response.source_entity_id == "entity-001"
    assert response.source_entity_name == "Port of Rotterdam"
    assert response.source_risk_score == 80.0
    assert response.affected_count == 2
    # Depth-1 entity at attenuation 0.5: 80 * 0.5 = 40 -> MEDIUM.
    first = next(e for e in response.affected_entities if e.entity_id == "entity-002")
    assert first.depth == 1
    assert first.propagated_risk_score == 40.0
    assert first.propagated_risk_level == "MEDIUM"
    # Depth-2 entity: 80 * 0.25 = 20 -> LOW.
    second = next(e for e in response.affected_entities if e.entity_id == "entity-003")
    assert second.depth == 2
    assert second.propagated_risk_score == 20.0
    assert second.propagated_risk_level == "LOW"


# --------------------------------------------------------------------------- #
# 2. Valid input accepted
# --------------------------------------------------------------------------- #


def test_valid_input(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record()
    mock_repository.find_downstream_neighbors.return_value = []

    response = service.propagate(_request())

    assert response.propagated is True
    assert response.affected_count == 0
    assert response.max_depth_reached == 0


# --------------------------------------------------------------------------- #
# 3. No downstream neighbors -> empty but propagated
# --------------------------------------------------------------------------- #


def test_no_downstream_neighbors(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record()
    mock_repository.find_downstream_neighbors.return_value = []

    response = service.propagate(_request())

    assert response.propagated is True
    assert response.affected_entities == []
    assert response.affected_count == 0


# --------------------------------------------------------------------------- #
# 4. Unresolved entity -> propagated=False, no DB access
# --------------------------------------------------------------------------- #


def test_unresolved_entity_skips_propagation(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    response = service.propagate(
        _request(entity_id=None, entity_name="Unresolved Place")
    )

    assert response.propagated is False
    assert response.affected_count == 0
    assert response.error is not None
    mock_repository.get_node_by_id.assert_not_called()
    mock_repository.find_downstream_neighbors.assert_not_called()


# --------------------------------------------------------------------------- #
# 5. Entity not found -> EntityNotFoundError, no nodes created
# --------------------------------------------------------------------------- #


def test_entity_not_found_raises(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = None

    with pytest.raises(EntityNotFoundError):
        service.propagate(_request())
    mock_repository.find_downstream_neighbors.assert_not_called()


# --------------------------------------------------------------------------- #
# 6. Neo4j unavailable -> ServiceUnavailable propagates
# --------------------------------------------------------------------------- #


def test_neo4j_unavailable_during_source_lookup(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.side_effect = ServiceUnavailable("Neo4j down")

    with pytest.raises(ServiceUnavailable):
        service.propagate(_request())


def test_neo4j_unavailable_during_traversal(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record()
    mock_repository.find_downstream_neighbors.side_effect = ServiceUnavailable(
        "Neo4j down"
    )

    with pytest.raises(ServiceUnavailable):
        service.propagate(_request())


# --------------------------------------------------------------------------- #
# 7. Repository failure -> Neo4jError propagates
# --------------------------------------------------------------------------- #


def test_repository_failure_propagates(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record()
    mock_repository.find_downstream_neighbors.side_effect = Neo4jError(
        "constraint violation"
    )

    with pytest.raises(Neo4jError):
        service.propagate(_request())


# --------------------------------------------------------------------------- #
# 8. Cycle prevention
# --------------------------------------------------------------------------- #


def test_cycle_prevention_deduplicates(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record()
    # The same downstream node would be reached in both hops; the service must
    # record it once, at the shallowest depth.
    mock_repository.find_downstream_neighbors.side_effect = [
        [_neighbor_record("entity-002", "Gigafactory Assembly")],
        [_neighbor_record("entity-002", "Gigafactory Assembly")],
    ]

    response = service.propagate(_request(max_depth=3))

    matches = [e for e in response.affected_entities if e.entity_id == "entity-002"]
    assert len(matches) == 1
    assert matches[0].depth == 1


def test_source_node_not_included_as_affected(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record()
    mock_repository.find_downstream_neighbors.side_effect = [
        [_neighbor_record("entity-001", "Port of Rotterdam")]
    ]

    response = service.propagate(_request())

    assert all(e.entity_id != "entity-001" for e in response.affected_entities)
    assert response.affected_count == 0


# --------------------------------------------------------------------------- #
# 9. Configurable depth / attenuation honoured
# --------------------------------------------------------------------------- #


def test_max_depth_limits_traversal(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record()
    mock_repository.find_downstream_neighbors.side_effect = [
        [_neighbor_record("entity-002", "Gigafactory Assembly")],
        [_neighbor_record("entity-003", "Regional Warehouse Hub")],
        [_neighbor_record("entity-004", "EV Distribution Center")],
    ]

    response = service.propagate(_request(max_depth=2))

    assert mock_repository.find_downstream_neighbors.call_count == 2
    assert response.max_depth_reached == 2
    assert response.affected_count == 2


def test_attenuation_zero_flattens_risk(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record(risk_score=80.0)
    mock_repository.find_downstream_neighbors.side_effect = [
        [_neighbor_record("entity-002", "Gigafactory Assembly")],
        [_neighbor_record("entity-003", "Regional Warehouse Hub")],
    ]

    response = service.propagate(_request(max_depth=2, attenuation=0.0))

    for entity in response.affected_entities:
        assert entity.propagated_risk_score == 0.0
        assert entity.propagated_risk_level == "LOW"


def test_persisted_risk_score_preferred_over_request(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record(risk_score=100.0)
    mock_repository.find_downstream_neighbors.return_value = [
        _neighbor_record("entity-002", "Gigafactory Assembly")
    ]

    response = service.propagate(_request(risk_score=40.0, attenuation=1.0))

    # The persisted Module 9 score (100) is used, not the request value (40).
    assert response.source_risk_score == 100.0
    assert response.affected_entities[0].propagated_risk_score == 100.0
    assert response.affected_entities[0].propagated_risk_level == "CRITICAL"


# --------------------------------------------------------------------------- #
# 10. Response structure & timestamp
# --------------------------------------------------------------------------- #


def test_response_structure(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record()
    mock_repository.find_downstream_neighbors.side_effect = [
        [_neighbor_record("entity-002", "Gigafactory Assembly")],
        [],
    ]

    response = service.propagate(_request())

    payload = response.model_dump()
    for field in (
        "source_entity_id",
        "source_entity_name",
        "source_risk_score",
        "propagated",
        "affected_entities",
        "affected_count",
        "max_depth_reached",
        "timestamp",
        "error",
    ):
        assert field in payload

    single = payload["affected_entities"][0]
    for field in (
        "entity_id",
        "entity_name",
        "depth",
        "propagated_risk_score",
        "propagated_risk_level",
    ):
        assert field in single


def test_timestamp_is_utc_aware(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record()
    mock_repository.find_downstream_neighbors.return_value = []

    response = service.propagate(_request())

    assert isinstance(response.timestamp, datetime)
    assert response.timestamp.tzinfo is not None


# --------------------------------------------------------------------------- #
# 11. Relationship types are forwarded to the repository
# --------------------------------------------------------------------------- #


def test_relationship_types_forwarded(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record()
    mock_repository.find_downstream_neighbors.return_value = []

    service.propagate(_request(relationship_types=["SUPPLIES"]))

    assert mock_repository.find_downstream_neighbors.call_args.args[0] == [
        "entity-001"
    ]
    assert mock_repository.find_downstream_neighbors.call_args.kwargs[
        "relationship_types"
    ] == ["SUPPLIES"]


# --------------------------------------------------------------------------- #
# 12. Propagation only reads the graph (never writes / creates nodes)
# --------------------------------------------------------------------------- #


def test_propagation_never_writes(
    service: RiskPropagationService, mock_repository: Mock
) -> None:
    mock_repository.get_node_by_id.return_value = _source_record()
    mock_repository.find_downstream_neighbors.return_value = []

    service.propagate(_request())

    mock_repository.get_node_by_id.assert_called_once()
    mock_repository.find_downstream_neighbors.assert_called_once()


# --------------------------------------------------------------------------- #
# Schema-level validation
# --------------------------------------------------------------------------- #


def test_schema_rejects_low_score() -> None:
    with pytest.raises(ValidationError):
        RiskPropagationRequest(entity_id="entity-001", risk_score=-5.0)


def test_schema_rejects_high_score() -> None:
    with pytest.raises(ValidationError):
        RiskPropagationRequest(entity_id="entity-001", risk_score=150.0)


def test_schema_rejects_zero_depth() -> None:
    with pytest.raises(ValidationError):
        RiskPropagationRequest(entity_id="entity-001", risk_score=50.0, max_depth=0)


def test_schema_rejects_out_of_range_attenuation() -> None:
    with pytest.raises(ValidationError):
        RiskPropagationRequest(
            entity_id="entity-001", risk_score=50.0, attenuation=1.5
        )


def test_schema_normalizes_blank_entity_id() -> None:
    request = RiskPropagationRequest(entity_id="   ", risk_score=50.0)
    assert request.entity_id is None


def test_schema_filters_blank_relationship_types() -> None:
    request = RiskPropagationRequest(
        entity_id="entity-001",
        risk_score=50.0,
        relationship_types=["", "  ", "SUPPLIES"],
    )
    assert request.relationship_types == ["SUPPLIES"]



