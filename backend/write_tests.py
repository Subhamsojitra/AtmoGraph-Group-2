"""Write the Part 4 WebSocket ripple prediction test file (part 1)."""

content = '''"""Tests for Module 17 Part 4: WebSocket -> RipplePredictionService integration.

These tests mount the real ``/api/v1/ws`` route but MOCK the ripple prediction
service at the WebSocket dependency, so NONE of them require a running Neo4j
server, a trained GNN checkpoint or a GPU. They verify the Module 17 request
dispatch, response normalization, error mapping and client isolation around
the EXISTING Module 17 service contract.

Conventions follow ``tests/test_websocket.py`` (Module 15),
``tests/test_websocket_prediction.py`` (Module 16) and
``tests/test_ripple_prediction_service.py`` (Module 17 Part 3).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Fake Neo4j env vars must be set before importing the app (pydantic-settings).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

from app.main import app  # noqa: E402
from app.schemas.websocket import (  # noqa: E402
    ERROR_INVALID_MESSAGE,
    ERROR_PREDICTION_FAILED,
    MESSAGE_TYPE_RIPPLE_PREDICTION_RESULT,
)
from app.services.ripple_prediction.exceptions import RipplePredictionError
from app.services.ripple_prediction.result_schema import (
    RippleAffectedEntity,
    RipplePredictionResult,
)
from app.services.risk.exceptions import EntityNotFoundError
from app.services.websocket_manager import reset_connection_manager  # noqa: E402

client = TestClient(app)
WS_URL = "/api/v1/ws"


@pytest.fixture(autouse=True)
def _clean_connection_manager() -> None:
    """Start every test with a fresh connection manager registry."""
    reset_connection_manager()
    yield
    reset_connection_manager()


@pytest.fixture()
def mock_ripple_service() -> MagicMock:
    """A RipplePredictionService mock used to isolate the WebSocket layer."""
    return MagicMock()


@pytest.fixture()
def api_client(mock_ripple_service: MagicMock) -> Any:
    """A base WebSocket client with the ripple service dependency overridden."""
    from app.api.websocket import get_ripple_prediction_service

    app.dependency_overrides[get_ripple_prediction_service] = (
        lambda: mock_ripple_service
    )
    yield client
    app.dependency_overrides.pop(get_ripple_prediction_service, None)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_ripple_result(
    source_entity_id: str = "SRC_001",
    source_entity_name: str = "Source Entity",
    source_risk_score: float = 75.0,
    affected_entities: list[RippleAffectedEntity] | None = None,
    propagated: bool = True,
    error: str | None = None,
) -> RipplePredictionResult:
    """Build a RipplePredictionResult with sensible defaults."""
    affected = affected_entities or [
        RippleAffectedEntity(
            entity_id="DST_001",
            entity_name="Downstream A",
            depth=1,
            propagated_risk_score=50.0,
            propagated_risk_level="MEDIUM",
            gnn_prediction=0.75,
        ),
    ]
    return RipplePredictionResult(
        source_entity_id=source_entity_id,
        source_entity_name=source_entity_name,
        source_risk_score=source_risk_score,
        propagated=propagated,
        affected_entities=affected,
        affected_count=len(affected),
        max_depth_reached=1,
        prediction_count=1,
        error=error,
    )
'''

with open(
    'D:/Infotact_projects/AtmoGraph-Group-2/backend/tests/'
    'test_websocket_ripple_prediction.py',
    'w',
) as f:
    f.write(content)

print('Part 1 written')
