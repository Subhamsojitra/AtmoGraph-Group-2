"""API tests for the Module 14 prediction endpoint (mocked service).

These tests inject a mocked :class:`PredictionService` through
``app.dependency_overrides`` so they do NOT require a running Neo4j server,
a trained checkpoint or a GPU. They verify the route exists, is registered,
appears in OpenAPI, and maps domain errors to the correct HTTP status codes
without leaking filesystem paths or stack traces.

Conventions follow ``tests/test_risk_propagation_api.py`` (Module 10).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

# Fake Neo4j env vars must be set before importing the app (pydantic-settings).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

from app.api.prediction import get_prediction_service  # noqa: E402
from app.main import app  # noqa: E402
from app.ml.exceptions import (  # noqa: E402
    EmptyGraphError,
    GraphDatasetValidationError,
    GNNCheckpointInvalidError,
    GNNModelIncompatibleError,
    GNNModelNotAvailableError,
    GNNPredictionRuntimeError,
)
from app.schemas.prediction import PredictionResponse  # noqa: E402
from app.services.prediction_service import ModelNotAvailableError  # noqa: E402

client = TestClient(app)
URL = "/api/v1/predictions"


@pytest.fixture()
def mock_service() -> MagicMock:
    """A PredictionService mock used to isolate the API layer."""
    return MagicMock()


@pytest.fixture()
def api_client(mock_service: MagicMock) -> Any:
    """A base client with the prediction service dependency overridden."""
    app.dependency_overrides[get_prediction_service] = lambda: mock_service
    yield client
    app.dependency_overrides.pop(get_prediction_service, None)


def _response(**overrides: Any) -> PredictionResponse:
    """Build a valid prediction response."""
    data: dict[str, Any] = {
        "predictions": [
            {"node_id": "entity-001", "prediction": 1.5},
            {"node_id": "entity-002", "prediction": 0.0},
        ],
        "prediction_count": 2,
    }
    data.update(overrides)
    return PredictionResponse(**data)


# ---------------------------------------------------------------------------
# 1. Success cases & request handling
# ---------------------------------------------------------------------------


def test_predict_success_returns_200_and_schema(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.return_value = _response()
    res = api_client.post(URL, json={})
    assert res.status_code == 200
    body = res.json()
    assert set(body.keys()) == {"predictions", "prediction_count", "timestamp"}
    assert body["prediction_count"] == 2
    assert len(body["predictions"]) == 2
    for entry in body["predictions"]:
        assert set(entry.keys()) == {"node_id", "prediction"}
        assert isinstance(entry["node_id"], str)
        assert isinstance(entry["prediction"], float)


def test_predict_without_body_succeeds(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.return_value = _response()
    res = api_client.post(URL)
    assert res.status_code == 200
    # The route builds a default request when the body is missing.
    request = mock_service.get_predictions.call_args.args[0]
    assert request.relationship_types is None


def test_predict_passes_relationship_types(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.return_value = _response()
    res = api_client.post(URL, json={"relationship_types": ["SUPPLIES"]})
    assert res.status_code == 200
    request = mock_service.get_predictions.call_args.args[0]
    assert request.relationship_types == ["SUPPLIES"]


def test_blank_relationship_types_are_cleaned(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.return_value = _response()
    res = api_client.post(
        URL, json={"relationship_types": ["SUPPLIES", "   "]}
    )
    assert res.status_code == 200
    request = mock_service.get_predictions.call_args.args[0]
    assert request.relationship_types == ["SUPPLIES"]


# ---------------------------------------------------------------------------
# 2. Invalid requests (Pydantic validation -> 422)
# ---------------------------------------------------------------------------


def test_relationship_types_wrong_type_returns_422(api_client: Any) -> None:
    res = api_client.post(URL, json={"relationship_types": "SUPPLIES"})
    assert res.status_code == 422


def test_non_object_body_returns_422(api_client: Any) -> None:
    res = api_client.post(URL, json=[1, 2, 3])
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# 3. Service errors -> HTTP status mapping (no internal detail leaks)
# ---------------------------------------------------------------------------


def test_model_not_configured_returns_503(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = ModelNotAvailableError(
        "no trained GNN checkpoint is configured"
    )
    res = api_client.post(URL, json={})
    assert res.status_code == 503
    assert "checkpoint" in res.json()["detail"].lower()
    assert ".pt" not in res.text


def test_missing_checkpoint_file_returns_503(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = GNNModelNotAvailableError(
        "trained GNN checkpoint not found: C:/secret/models/model.pt"
    )
    res = api_client.post(URL, json={})
    assert res.status_code == 503
    # No filesystem paths leak to clients.
    assert "C:" not in res.text
    assert ".pt" not in res.text


def test_neo4j_unavailable_returns_503(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = ServiceUnavailable("down")
    res = api_client.post(URL, json={})
    assert res.status_code == 503


def test_empty_graph_returns_503(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = EmptyGraphError("0 nodes")
    res = api_client.post(URL, json={})
    assert res.status_code == 503


def test_invalid_checkpoint_returns_500(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = GNNCheckpointInvalidError(
        "corrupt"
    )
    res = api_client.post(URL, json={})
    assert res.status_code == 500


def test_incompatible_model_returns_500(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = GNNModelIncompatibleError(
        "feature width"
    )
    res = api_client.post(URL, json={})
    assert res.status_code == 500


def test_inference_failure_returns_500(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = GNNPredictionRuntimeError("boom")
    res = api_client.post(URL, json={})
    assert res.status_code == 500


def test_dataset_validation_failure_returns_500(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = GraphDatasetValidationError(
        "bad data"
    )
    res = api_client.post(URL, json={})
    assert res.status_code == 500


def test_unexpected_error_returns_500(
    api_client: Any, mock_service: MagicMock
) -> None:
    mock_service.get_predictions.side_effect = RuntimeError("boom")
    res = api_client.post(URL, json={})
    assert res.status_code == 500
    assert res.json()["detail"] == "An unexpected internal error occurred."


# ---------------------------------------------------------------------------
# 4. Route registration
# ---------------------------------------------------------------------------


def test_route_is_registered_in_openapi() -> None:
    openapi = app.openapi()
    assert URL in openapi["paths"]
    assert "post" in openapi["paths"][URL]