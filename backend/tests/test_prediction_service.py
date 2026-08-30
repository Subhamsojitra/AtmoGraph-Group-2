"""Unit tests for the Module 14 prediction service (mocked dataset builder).

No Neo4j and no HTTP: the Module 11 dataset builder is replaced by a
synthetic stub so the service's orchestration (lazy reusable model
lifecycle, request passthrough, response mapping, error translation) is
verified in isolation. Checkpoints are tiny Module 13 artifacts trained on
[TEST/SYNTHETIC] data; no real-world accuracy is claimed anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Fake Neo4j env vars must be set before importing the app (pydantic-settings).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

import numpy as np
import pytest
import torch

from app.core.config import settings
from app.ml.dataset import GraphDataset
from app.ml.exceptions import GNNModelNotAvailableError
from app.ml.model import GNNModel
from app.ml.training import GNNTrainingConfig, GNNTrainer
from app.schemas.prediction import PredictionRequest
from app.services.prediction_service import (
    ModelNotAvailableError,
    PredictionService,
)

FEATURE_DIM = 5


def make_dataset(num_nodes: int = 8, *, labeled: bool = True) -> GraphDataset:
    """Small synthetic Module 11 GraphDataset (no Neo4j involved)."""
    rows = np.arange(num_nodes, dtype=np.float32).reshape(-1, 1)
    x = np.repeat((rows + 1.0) / float(num_nodes + 1), FEATURE_DIM, axis=1)
    edge_index = np.zeros((2, 0), dtype=np.int64)
    y = None
    if labeled:
        y = np.linspace(0.1, 0.1 + 0.1 * (num_nodes - 1), num_nodes).astype(
            np.float32
        )
    return GraphDataset(
        node_ids=tuple(f"n{i}" for i in range(num_nodes)),
        x=x.astype(np.float32),
        edge_index=edge_index,
        edge_rel_types=(),
        y=y,
        target_name="downstream_delay_days" if labeled else None,
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


def write_checkpoint(tmp_path: Path) -> Path:
    """Train a tiny model on [TEST/SYNTHETIC] data and save the checkpoint."""
    dataset = make_dataset(8)
    torch.manual_seed(42)
    model = GNNModel(
        input_dim=FEATURE_DIM, hidden_dim=8, num_layers=2, dropout=0.0
    )
    trainer = GNNTrainer(
        model,
        dataset,
        GNNTrainingConfig(
            epochs=2,
            learning_rate=0.02,
            weight_decay=0.0,
            seed=42,
            device="cpu",
        ),
    )
    trainer.train()
    path = tmp_path / "service.pt"
    trainer.save_checkpoint(path)
    return path


# ---------------------------------------------------------------------------
# Service orchestration
# ---------------------------------------------------------------------------


def test_service_returns_predictions(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path)
    dataset = make_dataset(8)
    service = PredictionService(
        checkpoint_path=str(path), dataset_builder=_StubBuilder(dataset)
    )
    response = service.get_predictions()
    assert response.prediction_count == 8
    assert [p.node_id for p in response.predictions] == list(dataset.node_ids)
    assert all(isinstance(p.prediction, float) for p in response.predictions)
    assert response.timestamp is not None


def test_service_passes_relationship_types_to_builder(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path)
    builder = _StubBuilder(make_dataset(8))
    service = PredictionService(
        checkpoint_path=str(path), dataset_builder=builder
    )
    service.get_predictions(PredictionRequest(relationship_types=["SUPPLIES"]))
    assert builder.calls == [["SUPPLIES"]]


def test_service_default_request_has_no_filters(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path)
    builder = _StubBuilder(make_dataset(8))
    service = PredictionService(
        checkpoint_path=str(path), dataset_builder=builder
    )
    service.get_predictions()
    assert builder.calls == [None]


def test_service_reuses_loaded_model(tmp_path: Path) -> None:
    """The model is loaded lazily ONCE and reused across predictions."""
    path = write_checkpoint(tmp_path)
    service = PredictionService(
        checkpoint_path=str(path),
        dataset_builder=_StubBuilder(make_dataset(8)),
    )
    service.get_predictions()
    first = service._predictor
    assert first is not None
    service.get_predictions()
    assert service._predictor is first


def test_service_reload_model_forces_new_predictor(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path)
    service = PredictionService(
        checkpoint_path=str(path),
        dataset_builder=_StubBuilder(make_dataset(8)),
    )
    service.get_predictions()
    first = service._predictor
    service.reload_model()
    service.get_predictions()
    assert service._predictor is not first


def test_service_without_configured_checkpoint_raises(monkeypatch) -> None:
    monkeypatch.setattr(settings, "prediction_checkpoint_path", None)
    service = PredictionService(dataset_builder=_StubBuilder(make_dataset(8)))
    with pytest.raises(ModelNotAvailableError, match="checkpoint"):
        service.get_predictions()


def test_service_uses_settings_checkpoint_path(tmp_path: Path, monkeypatch) -> None:
    path = write_checkpoint(tmp_path)
    monkeypatch.setattr(settings, "prediction_checkpoint_path", str(path))
    service = PredictionService(dataset_builder=_StubBuilder(make_dataset(8)))
    response = service.get_predictions()
    assert response.prediction_count == 8


def test_service_missing_checkpoint_file_raises(tmp_path: Path) -> None:
    service = PredictionService(
        checkpoint_path=str(tmp_path / "missing.pt"),
        dataset_builder=_StubBuilder(make_dataset(8)),
    )
    with pytest.raises(GNNModelNotAvailableError):
        service.get_predictions()


def test_service_device_and_path_overrides(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path)
    service = PredictionService(
        checkpoint_path=str(path),
        device="cpu",
        dataset_builder=_StubBuilder(make_dataset(8)),
    )
    assert service.device == "cpu"
    assert service.checkpoint_path == path


def test_service_checkpoint_path_defaults_to_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "prediction_checkpoint_path", None)
    service = PredictionService()
    assert service.checkpoint_path is None