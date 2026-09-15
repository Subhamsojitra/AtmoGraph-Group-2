"""Integration tests: REAL Module 13 checkpoint -> REAL PredictionService.

Unlike ``tests/test_websocket_prediction.py`` (which mocks the service at the
WebSocket dependency), these tests prove the ACTUAL trained-checkpoint
inference path end to end:

    Module 13 GNNTrainer (real training on [TEST/SYNTHETIC] data)
        -> real checkpoint file on disk
        -> PredictionService (real GNNPredictor load,
           torch.load(..., weights_only=True))
        -> real GNN inference (eval mode, torch.no_grad)
        -> WebSocket prediction_request -> prediction_result

The Module 11 dataset builder is replaced by the established synthetic stub
(same pattern as ``tests/test_prediction_service.py``) because the suite runs
without a Neo4j server. NO prediction values are mocked: every number asserted
below comes from the real model's forward pass over the real checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# Fake Neo4j env vars must be set before importing the app (pydantic-settings).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from neo4j.exceptions import ServiceUnavailable  # noqa: E402

from app.core.config import BACKEND_DIR, settings  # noqa: E402
from app.main import app  # noqa: E402
from app.ml.dataset import GraphDataset  # noqa: E402
from app.ml.exceptions import (  # noqa: E402
    GNNCheckpointInvalidError,
    GNNModelNotAvailableError,
    GNNPredictionRuntimeError,
)
from app.ml.model import GNNModel  # noqa: E402
from app.ml.training import GNNTrainingConfig, GNNTrainer  # noqa: E402
from app.schemas.websocket import (  # noqa: E402
    ERROR_MODEL_UNAVAILABLE,
    ERROR_NODE_NOT_FOUND,
    ERROR_PREDICTION_FAILED,
    MESSAGE_TYPE_PREDICTION_RESULT,
)
from app.services.prediction_service import (  # noqa: E402
    ModelNotAvailableError,
    PredictionService,
)
from app.services.websocket_manager import (  # noqa: E402
    reset_connection_manager,
)
from app.services.websocket_prediction import (  # noqa: E402
    get_websocket_prediction_service,
)

client = TestClient(app)
WS_URL = "/api/v1/ws"
FEATURE_DIM = 5  # len(NodeFeatureEncoder.FEATURE_NAMES)
NUM_NODES = 16


def make_dataset(num_nodes: int = NUM_NODES) -> GraphDataset:
    """Small synthetic Module 11 GraphDataset (no Neo4j involved)."""
    rows = np.arange(num_nodes, dtype=np.float32).reshape(-1, 1)
    x = np.repeat((rows + 1.0) / float(num_nodes + 1), FEATURE_DIM, axis=1)
    edge_index = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    y = np.linspace(0.1, 0.1 + 0.1 * (num_nodes - 1), num_nodes).astype(
        np.float32
    )
    return GraphDataset(
        node_ids=tuple(f"n{i}" for i in range(num_nodes)),
        x=x.astype(np.float32),
        edge_index=edge_index,
        edge_rel_types=("SUPPLIES", "SUPPLIES", "SUPPLIES"),
        y=y,
        target_name="downstream_delay_days",
        metadata=None,
    )


class _StubBuilder:
    """Replaces the Module 11 GraphDatasetBuilder (no Neo4j access)."""

    def __init__(self, dataset: GraphDataset) -> None:
        self.dataset = dataset
        self.calls: list[Optional[list[str]]] = []

    def build(self, relationship_types: Optional[list[str]] = None) -> GraphDataset:
        self.calls.append(relationship_types)
        return self.dataset


class _FailingBuilder:
    """Builder stub whose build() always raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def build(self, relationship_types: Optional[list[str]] = None) -> GraphDataset:
        raise self.exc


def node_ids() -> list[str]:
    return [f"n{i}" for i in range(NUM_NODES)]


@pytest.fixture(scope="module")
def real_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train ONE tiny REAL Module 13 checkpoint on [TEST/SYNTHETIC] data."""
    dataset = make_dataset()
    torch.manual_seed(42)
    model = GNNModel(
        input_dim=FEATURE_DIM, hidden_dim=8, num_layers=2, dropout=0.0
    )
    trainer = GNNTrainer(
        model,
        dataset,
        GNNTrainingConfig(
            epochs=3,
            learning_rate=0.02,
            weight_decay=0.0,
            seed=42,
            device="cpu",
        ),
    )
    trainer.train()
    path = tmp_path_factory.mktemp("real_checkpoint") / "real.pt"
    trainer.save_checkpoint(path)
    return path


@pytest.fixture()
def real_service(
    real_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> PredictionService:
    """A REAL PredictionService: real checkpoint, stub dataset builder."""
    monkeypatch.setattr(
        settings, "prediction_checkpoint_path", str(real_checkpoint)
    )
    return PredictionService(dataset_builder=_StubBuilder(make_dataset()))


@pytest.fixture(autouse=True)
def _clean_connection_manager() -> None:
    """Start every test with a fresh connection manager registry."""
    reset_connection_manager()
    yield
    reset_connection_manager()


@pytest.fixture()
def api_client(real_service: PredictionService) -> Any:
    """WebSocket client wired to the REAL PredictionService."""
    app.dependency_overrides[get_websocket_prediction_service] = (
        lambda: real_service
    )
    yield client
    app.dependency_overrides.pop(get_websocket_prediction_service, None)


# --------------------------------------------------------------------------- #
# 1. Checkpoint configuration resolution
# --------------------------------------------------------------------------- #


def test_relative_settings_checkpoint_path_resolves_against_backend_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative PREDICTION_CHECKPOINT_PATH is backend-relative, not CWD-relative."""
    monkeypatch.setattr(
        settings, "prediction_checkpoint_path", "checkpoints/gnn_m13.pt"
    )
    service = PredictionService()
    assert service.checkpoint_path == (
        BACKEND_DIR / "checkpoints" / "gnn_m13.pt"
    )


def test_relative_override_checkpoint_path_resolves_against_backend_dir() -> None:
    service = PredictionService(checkpoint_path="checkpoints/relative.pt")
    assert service.checkpoint_path == (
        BACKEND_DIR / "checkpoints" / "relative.pt"
    )


def test_absolute_checkpoint_path_is_used_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "abs.pt"
    service = PredictionService(checkpoint_path=str(target))
    assert service.checkpoint_path == target


def test_no_checkpoint_configured_raises_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No configured checkpoint stays a clean deployment state, not a crash."""
    monkeypatch.setattr(settings, "prediction_checkpoint_path", None)
    service = PredictionService(dataset_builder=_StubBuilder(make_dataset()))
    with pytest.raises(ModelNotAvailableError, match="checkpoint"):
        service.get_predictions()


# --------------------------------------------------------------------------- #
# 2. Real checkpoint loading & real inference (service level)
# --------------------------------------------------------------------------- #


def test_real_checkpoint_loads_and_serves_real_predictions(
    real_checkpoint: Path,
) -> None:
    service = PredictionService(
        checkpoint_path=str(real_checkpoint),
        dataset_builder=_StubBuilder(make_dataset()),
    )
    response = service.get_predictions()

    assert service.checkpoint_path == real_checkpoint
    assert response.prediction_count == NUM_NODES
    assert [p.node_id for p in response.predictions] == node_ids()
    values = [p.prediction for p in response.predictions]
    # Real model output: finite floats, no mocked/fabricated values anywhere.
    assert all(isinstance(v, float) for v in values)
    assert all(np.isfinite(values))

    # Eval-mode inference is deterministic: the same dataset yields the same
    # REAL predictions, and the loaded model is reused (loaded once).
    first = service._predictor
    again = service.get_predictions()
    assert [p.prediction for p in again.predictions] == values
    assert service._predictor is first


def test_real_checkpoint_matches_the_dataset_feature_width(
    real_checkpoint: Path,
) -> None:
    """Guard: the loaded model config agrees with the served dataset."""
    service = PredictionService(
        checkpoint_path=str(real_checkpoint),
        dataset_builder=_StubBuilder(make_dataset()),
    )
    predictor = service._get_predictor()
    assert predictor.model_config.input_dim == FEATURE_DIM
    assert predictor.model_config.output_dim == 1
    # The auto-saved checkpoint carries the correct training provenance
    # (the fixture trains for exactly 3 epochs).
    assert predictor._checkpoint.epoch == 3
    assert predictor._checkpoint.best_val_loss is not None


def test_corrupt_checkpoint_raises_structured_error(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.pt"
    torch.save({"unrelated": [1, 2, 3]}, bad)
    service = PredictionService(
        checkpoint_path=str(bad),
        dataset_builder=_StubBuilder(make_dataset()),
    )
    with pytest.raises(GNNCheckpointInvalidError):
        service.get_predictions()


def test_missing_checkpoint_file_raises_model_not_available(
    tmp_path: Path,
) -> None:
    service = PredictionService(
        checkpoint_path=str(tmp_path / "missing.pt"),
        dataset_builder=_StubBuilder(make_dataset()),
    )
    with pytest.raises(GNNModelNotAvailableError):
        service.get_predictions()


# --------------------------------------------------------------------------- #
# 3. WebSocket end-to-end over the REAL checkpoint path
# --------------------------------------------------------------------------- #


def _whole_graph_values(service: PredictionService) -> dict[str, float]:
    return {
        p.node_id: p.prediction for p in service.get_predictions().predictions
    }


def test_websocket_prediction_request_returns_real_model_output(
    api_client: Any, real_service: PredictionService
) -> None:
    expected = _whole_graph_values(real_service)
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "prediction_request"})
        result = websocket.receive_json()
    assert result["type"] == MESSAGE_TYPE_PREDICTION_RESULT
    data = result["data"]
    assert data["prediction_count"] == NUM_NODES
    assert [p["node_id"] for p in data["predictions"]] == node_ids()
    # The values served over the WebSocket ARE the real model's output.
    for entry in data["predictions"]:
        assert entry["prediction"] == expected[entry["node_id"]]


def test_websocket_single_node_request_returns_selected_real_value(
    api_client: Any, real_service: PredictionService
) -> None:
    expected = _whole_graph_values(real_service)
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "prediction_request", "data": {"node_id": "n5"}}
        )
        result = websocket.receive_json()
    assert result["type"] == MESSAGE_TYPE_PREDICTION_RESULT
    data = result["data"]
    assert data["requested_node_id"] == "n5"
    assert data["prediction_count"] == 1
    assert data["predictions"] == [
        {"node_id": "n5", "prediction": expected["n5"]}
    ]


def test_websocket_invalid_node_id_returns_node_not_found_and_stays_usable(
    api_client: Any,
) -> None:
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json(
            {"type": "prediction_request", "data": {"node_id": "ghost-node"}}
        )
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == ERROR_NODE_NOT_FOUND
        assert "ghost-node" in error["error"]["message"]
        # The connection remains fully usable after the failed request.
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_websocket_ping_pong_around_real_prediction(api_client: Any) -> None:
    with api_client.websocket_connect(WS_URL) as websocket:
        websocket.receive_json()  # consume connected
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"
        websocket.send_json({"type": "prediction_request"})
        assert websocket.receive_json()["type"] == MESSAGE_TYPE_PREDICTION_RESULT
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_websocket_model_unavailable_when_no_checkpoint_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a configured checkpoint the REAL service answers MODEL_UNAVAILABLE."""
    monkeypatch.setattr(settings, "prediction_checkpoint_path", None)
    unconfigured = PredictionService(dataset_builder=_StubBuilder(make_dataset()))
    app.dependency_overrides[get_websocket_prediction_service] = (
        lambda: unconfigured
    )
    try:
        with client.websocket_connect(WS_URL) as websocket:
            websocket.receive_json()  # consume connected
            websocket.send_json({"type": "prediction_request"})
            error = websocket.receive_json()
            assert error["error"]["code"] == ERROR_MODEL_UNAVAILABLE
            assert "configured" in error["error"]["message"].lower()
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"
    finally:
        app.dependency_overrides.pop(get_websocket_prediction_service, None)


def test_websocket_neo4j_unavailable_returns_distinct_client_safe_message(
    real_checkpoint: Path,
) -> None:
    """A database outage is reported as such, never as a model failure."""
    service = PredictionService(
        checkpoint_path=str(real_checkpoint),
        dataset_builder=_FailingBuilder(ServiceUnavailable("Neo4j down")),
    )
    app.dependency_overrides[get_websocket_prediction_service] = lambda: service
    try:
        with client.websocket_connect(WS_URL) as websocket:
            websocket.receive_json()  # consume connected
            websocket.send_json({"type": "prediction_request"})
            error = websocket.receive_json()
            assert error["error"]["code"] == ERROR_PREDICTION_FAILED
            assert "graph database" in error["error"]["message"]
            # No internals leak to the client.
            assert "Neo4j down" not in error["error"]["message"]
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"
    finally:
        app.dependency_overrides.pop(get_websocket_prediction_service, None)


def test_websocket_corrupt_checkpoint_returns_structured_error_without_crash(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "corrupt.pt"
    torch.save({"unrelated": [1, 2, 3]}, bad)
    service = PredictionService(
        checkpoint_path=str(bad),
        dataset_builder=_StubBuilder(make_dataset()),
    )
    app.dependency_overrides[get_websocket_prediction_service] = lambda: service
    try:
        with client.websocket_connect(WS_URL) as websocket:
            websocket.receive_json()  # consume connected
            websocket.send_json({"type": "prediction_request"})
            error = websocket.receive_json()
            assert error["error"]["code"] == ERROR_PREDICTION_FAILED
            assert str(bad) not in error["error"]["message"]
            assert ".pt" not in error["error"]["message"]
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"
    finally:
        app.dependency_overrides.pop(get_websocket_prediction_service, None)


def test_websocket_inference_failure_returns_structured_error(
    real_checkpoint: Path,
) -> None:
    service = PredictionService(
        checkpoint_path=str(real_checkpoint),
        dataset_builder=_FailingBuilder(GNNPredictionRuntimeError("boom")),
    )
    app.dependency_overrides[get_websocket_prediction_service] = lambda: service
    try:
        with client.websocket_connect(WS_URL) as websocket:
            websocket.receive_json()  # consume connected
            websocket.send_json({"type": "prediction_request"})
            error = websocket.receive_json()
            assert error["error"]["code"] == ERROR_PREDICTION_FAILED
            assert "boom" not in error["error"]["message"]
            assert "Traceback" not in error["error"]["message"]
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"
    finally:
        app.dependency_overrides.pop(get_websocket_prediction_service, None)


def test_websocket_multiple_clients_isolated_with_real_service(
    api_client: Any, real_service: PredictionService
) -> None:
    """Two concurrent clients each get their own REAL, consistent results."""
    expected = _whole_graph_values(real_service)
    with api_client.websocket_connect(WS_URL) as ws1:
        ws1.receive_json()  # consume connected
        with api_client.websocket_connect(WS_URL) as ws2:
            ws2.receive_json()  # consume connected
            ws1.send_json({"type": "prediction_request"})
            result1 = ws1.receive_json()
            ws2.send_json(
                {"type": "prediction_request", "data": {"node_id": "n0"}}
            )
            result2 = ws2.receive_json()

    assert result1["type"] == MESSAGE_TYPE_PREDICTION_RESULT
    assert result2["type"] == MESSAGE_TYPE_PREDICTION_RESULT
    for entry in result1["data"]["predictions"]:
        assert entry["prediction"] == expected[entry["node_id"]]
    assert result2["data"]["requested_node_id"] == "n0"
    assert result2["data"]["predictions"] == [
        {"node_id": "n0", "prediction": expected["n0"]}
    ]
