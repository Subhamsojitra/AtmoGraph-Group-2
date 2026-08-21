"""Unit tests for the risk service layer (Module 9).

These tests mock the :class:`GraphRepository` so they do NOT require a running
Neo4j server. Every test exercises the :class:`RiskService` business logic in
isolation.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import pytest
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import ValidationError

from app.repositories.graph_repository import GraphRepository
from app.schemas.risk import RiskLevelUpdateRequest, RiskStateUpdateResponse
from app.services.risk.exceptions import EntityNotFoundError
from app.services.risk.risk_service import RiskService


def _update_record(
    node_id: str = "entity-001",
    name: str = "Port of Rotterdam",
    prev_score: float = 30.0,
    prev_level: str = "LOW",
) -> list[dict[str, object]]:
    """Build the plain-dict record shape that mirrors a Neo4j query result."""
    return [
        {
            "n": {
                "id": node_id,
                "name": name,
                "risk_score": prev_score,
                "risk_level": prev_level,
            },
            "prev_score": prev_score,
            "prev_level": prev_level,
        }
    ]


def _request(
    risk_score: float = 65.0,
    entity_id: str = "entity-001",
    entity_name: str = "Port of Rotterdam",
    reason: str = "Port disruption",
) -> RiskLevelUpdateRequest:
    """Build a valid risk update request."""
    return RiskLevelUpdateRequest(
        entity_id=entity_id,
        entity_name=entity_name,
        risk_score=risk_score,
        reason=reason,
    )


@pytest.fixture()
def mock_repository() -> Mock:
    """A GraphRepository mock used to isolate the service from Neo4j."""
    return Mock(spec=GraphRepository)


@pytest.fixture()
def service(mock_repository: Mock) -> RiskService:
    """A RiskService wired to the mocked repository."""
    return RiskService(repository=mock_repository)

# --------------------------------------------------------------------------- #
# 1. Successful risk update
# --------------------------------------------------------------------------- #


def test_successful_risk_update(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = _update_record(prev_score=30.0, prev_level="LOW")

    response = service.update_risk_state(_request(risk_score=85.0))

    assert response.updated is True
    assert response.entity_id == "entity-001"
    assert response.entity_name == "Port of Rotterdam"
    assert response.new_risk_score == 85.0
    assert response.new_risk_level == "HIGH"
    assert response.previous_risk_score == 30.0
    assert response.previous_risk_level == "LOW"
    mock_repository.execute_write.assert_called_once()


def test_valid_risk_score(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = _update_record()

    response = service.update_risk_state(_request(risk_score=50.0))

    assert response.updated is True
    assert response.new_risk_score == 50.0
    assert response.new_risk_level == "MEDIUM"


def test_invalid_low_score_raises(service: RiskService) -> None:
    # ``model_construct`` bypasses the schema's ge/le so the service's own
    # validation is exercised rather than Pydantic's.
    request = RiskLevelUpdateRequest.model_construct(entity_id="entity-001", risk_score=-1.0)
    with pytest.raises(ValueError, match="risk_score must be between"):
        service.update_risk_state(request)


def test_invalid_high_score_raises(service: RiskService) -> None:
    request = RiskLevelUpdateRequest.model_construct(entity_id="entity-001", risk_score=101.0)
    with pytest.raises(ValueError, match="risk_score must be between"):
        service.update_risk_state(request)


def test_lower_boundary_accepted(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = _update_record()

    response = service.update_risk_state(_request(risk_score=0.0))

    assert response.updated is True
    assert response.new_risk_score == 0.0
    assert response.new_risk_level == "LOW"


def test_upper_boundary_accepted(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = _update_record()

    response = service.update_risk_state(_request(risk_score=100.0))

    assert response.updated is True
    assert response.new_risk_score == 100.0
    assert response.new_risk_level == "CRITICAL"


@pytest.mark.parametrize(
    ("score", "expected_level"),
    [
        (0.0, "LOW"),
        (30.0, "LOW"),
        (31.0, "MEDIUM"),
        (70.0, "MEDIUM"),
        (71.0, "HIGH"),
        (90.0, "HIGH"),
        (91.0, "CRITICAL"),
        (100.0, "CRITICAL"),
    ],
)
def test_risk_level_calculation(service: RiskService, score: float, expected_level: str) -> None:
    assert service.calculate_risk_level(score) == expected_level


def test_existing_risk_state_report(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = _update_record(prev_score=30.0, prev_level="LOW")

    response = service.update_risk_state(_request(risk_score=85.0))

    assert response.previous_risk_score == 30.0
    assert response.previous_risk_level == "LOW"
    assert response.new_risk_score == 85.0
    assert response.new_risk_level == "HIGH"


def test_missing_previous_risk_state_captured_as_none(
    service: RiskService, mock_repository: Mock
) -> None:
    mock_repository.execute_write.return_value = [
        {
            "n": {"id": "entity-001", "name": "Port of Rotterdam"},
            "prev_score": None,
            "prev_level": None,
        }
    ]

    response = service.update_risk_state(_request(risk_score=85.0))

    assert response.previous_risk_score is None
    assert response.previous_risk_level is None
    assert response.updated is True


# --------------------------------------------------------------------------- #
# 9. Unresolved entity -> no DB write
# --------------------------------------------------------------------------- #


def test_unresolved_entity_skips_update(service: RiskService, mock_repository: Mock) -> None:
    # No entity_id => unresolved (e.g. Module 8 returned matched=false).
    response = service.update_risk_state(RiskLevelUpdateRequest(risk_score=85.0))

    assert response.updated is False
    assert response.error is not None
    assert "unresolved" in response.error.lower()
    mock_repository.execute_write.assert_not_called()


def test_blank_entity_id_is_unresolved(service: RiskService, mock_repository: Mock) -> None:
    response = service.update_risk_state(
        RiskLevelUpdateRequest(entity_id="   ", risk_score=85.0)
    )

    assert response.updated is False
    mock_repository.execute_write.assert_not_called()


# --------------------------------------------------------------------------- #
# 10. Missing node -> EntityNotFoundError (no node created)
# --------------------------------------------------------------------------- #


def test_missing_node_raises_not_found(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = []

    with pytest.raises(EntityNotFoundError):
        service.update_risk_state(_request())


# --------------------------------------------------------------------------- #
# 11. Repository failure -> Neo4jError propagates
# --------------------------------------------------------------------------- #


def test_repository_failure_propagates(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.side_effect = Neo4jError("constraint violation")

    with pytest.raises(Neo4jError):
        service.update_risk_state(_request())


# --------------------------------------------------------------------------- #
# 12. Neo4j unavailable -> ServiceUnavailable propagates (503 upstream)
# --------------------------------------------------------------------------- #


def test_neo4j_unavailable_propagates(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.side_effect = ServiceUnavailable("Neo4j down")

    with pytest.raises(ServiceUnavailable):
        service.update_risk_state(_request())


# --------------------------------------------------------------------------- #
# 13. Response structure
# --------------------------------------------------------------------------- #


def test_response_structure(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = _update_record()

    response = service.update_risk_state(_request(reason="Port disruption"))

    assert isinstance(response, RiskStateUpdateResponse)
    payload = response.model_dump()
    for field in (
        "entity_id",
        "entity_name",
        "updated",
        "previous_risk_score",
        "new_risk_score",
        "previous_risk_level",
        "new_risk_level",
        "reason",
        "timestamp",
        "error",
    ):
        assert field in payload


# --------------------------------------------------------------------------- #
# 14. Timestamp
# --------------------------------------------------------------------------- #


def test_timestamp_is_utc_aware(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = _update_record()

    response = service.update_risk_state(_request())

    assert isinstance(response.timestamp, datetime)
    assert response.timestamp.tzinfo is not None

# --------------------------------------------------------------------------- #
# 15. Reason
# --------------------------------------------------------------------------- #


def test_reason_is_echoed(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = _update_record()

    response = service.update_risk_state(_request(reason="Port disruption"))

    assert response.reason == "Port disruption"


def test_reason_optional(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = _update_record()

    response = service.update_risk_state(_request(reason=None))

    assert response.reason is None


# --------------------------------------------------------------------------- #
# 16. No node creation
# --------------------------------------------------------------------------- #


def test_update_query_never_creates_nodes(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = _update_record()

    service.update_risk_state(_request())

    query = mock_repository.execute_write.call_args.args[0]
    assert "CREATE" not in query.upper()
    assert "MERGE" not in query.upper()
    assert "MATCH" in query.upper()


# --------------------------------------------------------------------------- #
# 17. Parameterized query behaviour (no interpolation of user input)
# --------------------------------------------------------------------------- #


def test_update_query_is_parameterized(service: RiskService, mock_repository: Mock) -> None:
    mock_repository.execute_write.return_value = _update_record()

    service.update_risk_state(_request(entity_id="entity-001", risk_score=65.0))

    query, parameters = mock_repository.execute_write.call_args.args
    assert "$entity_id" in query
    assert "$risk_score" in query
    assert "$risk_level" in query
    # The raw user value must never be interpolated into the query string.
    assert "entity-001" not in query
    assert parameters == {"entity_id": "entity-001", "risk_score": 65.0, "risk_level": "MEDIUM"}


# --------------------------------------------------------------------------- #
# Schema-level validation
# --------------------------------------------------------------------------- #


def test_schema_rejects_low_score() -> None:
    with pytest.raises(ValidationError):
        RiskLevelUpdateRequest(entity_id="entity-001", risk_score=-5.0)


def test_schema_rejects_high_score() -> None:
    with pytest.raises(ValidationError):
        RiskLevelUpdateRequest(entity_id="entity-001", risk_score=150.0)


def test_schema_normalizes_blank_entity_id() -> None:
    request = RiskLevelUpdateRequest(entity_id="   ", risk_score=50.0)
    assert request.entity_id is None
