"""Unit tests for the GNN prediction / inference layer (Module 14).

Fast, deterministic, CPU-only tests around ``app.ml.prediction``. Every
fixture below is explicitly [TEST/SYNTHETIC]: no Neo4j, no internet, no GPU,
no downloads and no accuracy claims. The trained checkpoints used here come
from tiny Module 13 training runs on synthetic data — they prove that the
inference layer LOADS, VALIDATES, MAPS and SERVES correctly; they do NOT
claim real-world predictive accuracy (the project database contains no real
downstream-delay labels).

Conventions follow ``tests/test_gnn_training.py`` (Module 13).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from app.ml.dataset import GraphDataset
from app.ml.exceptions import (
    GNNCheckpointInvalidError,
    GNNModelIncompatibleError,
    GNNModelNotAvailableError,
    GNNPredictionInputError,
    GNNPredictionRuntimeError,
)
from app.ml.model import GNNConfig, GNNModel
from app.ml.prediction import GNNPredictionResult, GNNPredictor
from app.ml.training import GNNTrainingConfig, GNNTrainer


# ---------------------------------------------------------------------------
# [TEST/SYNTHETIC] Deterministic fixtures - every value here is invented.
# ---------------------------------------------------------------------------

FEATURE_DIM = 5


def make_x(num_nodes: int, feature_dim: int = FEATURE_DIM) -> np.ndarray:
    """Deterministic feature rows that DIFFER per node (row i encodes i)."""
    rows = np.arange(num_nodes, dtype=np.float32).reshape(-1, 1)
    multipliers = np.arange(1, feature_dim + 1, dtype=np.float32)
    values = (rows + 1.0) * multipliers / float(feature_dim * (num_nodes + 1))
    return values.astype(np.float32)


def chain_edge_index(num_nodes: int) -> np.ndarray:
    """0 -> 1 -> ... -> N-1 (upstream flows downstream, Module 11 direction)."""
    if num_nodes < 2:
        return np.zeros((2, 0), dtype=np.int64)
    sources = np.arange(num_nodes - 1, dtype=np.int64)
    return np.stack([sources, sources + 1], axis=0)


def make_dataset(
    num_nodes: int = 8,
    *,
    labeled: bool = True,
    feature_dim: int = FEATURE_DIM,
) -> GraphDataset:
    """Small labeled (or unlabeled) Module 11 GraphDataset from synthetic values."""
    y = None
    if labeled:
        y = np.linspace(0.1, 0.1 + 0.1 * (num_nodes - 1), num_nodes).astype(
            np.float32
        )
    edge_index = chain_edge_index(num_nodes)
    return GraphDataset(
        node_ids=tuple(f"n{i}" for i in range(num_nodes)),
        x=make_x(num_nodes, feature_dim=feature_dim),
        edge_index=edge_index,
        edge_rel_types=("SUPPLIES",) * edge_index.shape[1],
        y=y,
        target_name="downstream_delay_days" if labeled else None,
        metadata=None,
    )


def make_model(
    *,
    input_dim: int = FEATURE_DIM,
    hidden_dim: int = 8,
    dropout: float = 0.0,
    output_dim: int = 1,
) -> GNNModel:
    """Small deterministic (initially untrained) Module 12 model."""
    torch.manual_seed(42)
    return GNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        output_dim=output_dim,
        dropout=dropout,
    )


def make_config(**overrides: object) -> GNNTrainingConfig:
    """Fast deterministic default training config (3 epochs, seeded, CPU)."""
    values: dict = {
        "epochs": 3,
        "learning_rate": 0.02,
        "weight_decay": 0.0,
        "seed": 42,
        "device": "cpu",
        "validation_split": 0.2,
        "test_split": 0.1,
        "eval_interval": 1,
    }
    values.update(overrides)
    return GNNTrainingConfig(**values)


def train_and_save(
    tmp_path: Path,
    *,
    num_nodes: int = 8,
    hidden_dim: int = 8,
    dropout: float = 0.0,
    filename: str = "m14.pt",
) -> tuple[Path, GNNTrainer, GraphDataset]:
    """Train a tiny model on synthetic data and save a Module 13 checkpoint."""
    dataset = make_dataset(num_nodes)
    model = make_model(hidden_dim=hidden_dim, dropout=dropout)
    trainer = GNNTrainer(model, dataset, make_config())
    trainer.train()
    path = tmp_path / filename
    trainer.save_checkpoint(path)
    return path, trainer, dataset


def write_manual_checkpoint(
    path: Path, *, model_config: GNNConfig, state_dict: dict
) -> Path:
    """Write a hand-crafted Module 13 checkpoint payload (for error tests)."""
    payload = {
        "format_version": 1,
        "model_state_dict": state_dict,
        "optimizer_state_dict": None,
        "epoch": 4,
        "best_val_loss": 0.5,
        "training_config": GNNTrainingConfig().to_dict(),
        "model_config": model_config.to_dict(),
        "metrics": None,
        "history": None,
    }
    torch.save(payload, path)
    return path


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Path:
    """One trained checkpoint shared by read-only inference tests (fast)."""
    path, _, _ = train_and_save(
        tmp_path_factory.mktemp("m14_ckpt"), filename="shared.pt"
    )
    return path


@pytest.fixture()
def predictor(trained: Path) -> GNNPredictor:
    """A predictor over the shared trained checkpoint (CPU)."""
    return GNNPredictor(trained, device="cpu")


# ---------------------------------------------------------------------------
# 1. Checkpoint loading & model configuration compatibility
# ---------------------------------------------------------------------------


def test_predictor_loads_trained_checkpoint(trained: Path) -> None:
    predictor = GNNPredictor(trained, device="cpu")
    assert predictor.model_config.input_dim == FEATURE_DIM
    assert predictor.model_config.output_dim == 1
    assert predictor.epoch >= 1
    assert predictor.checkpoint_path == trained
    # Inference-only: the loaded model must be in eval mode.
    assert predictor.model.training is False


def test_model_config_matches_checkpoint(trained: Path) -> None:
    predictor = GNNPredictor(trained, device="cpu")
    assert predictor.model.config == predictor.model_config
    assert predictor.model_config.hidden_dim == 8
    assert predictor.model_config.num_layers == 2


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(GNNModelNotAvailableError, match="not found"):
        GNNPredictor(tmp_path / "missing.pt")


def test_foreign_payload_checkpoint_rejected(tmp_path: Path) -> None:
    path = tmp_path / "foreign.pt"
    torch.save({"unrelated": [1, 2, 3]}, path)
    with pytest.raises(GNNCheckpointInvalidError):
        GNNPredictor(path)


def test_corrupt_checkpoint_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"this is not a torch checkpoint at all")
    with pytest.raises(GNNCheckpointInvalidError):
        GNNPredictor(path)


def test_output_dim_not_one_rejected(tmp_path: Path) -> None:
    """The serving contract is scalar per-node regression (output_dim == 1)."""
    model = make_model(output_dim=2)
    config = GNNConfig(
        input_dim=FEATURE_DIM,
        hidden_dim=8,
        num_layers=2,
        output_dim=2,
        dropout=0.0,
    )
    path = write_manual_checkpoint(
        tmp_path / "multi.pt", model_config=config, state_dict=model.state_dict()
    )
    with pytest.raises(GNNModelIncompatibleError, match="output_dim"):
        GNNPredictor(path)


def test_weight_config_mismatch_rejected(tmp_path: Path) -> None:
    """Weights from a hidden_dim=16 model cannot load into a hidden_dim=8 config."""
    big_model = make_model(hidden_dim=16)
    config = GNNConfig(
        input_dim=FEATURE_DIM,
        hidden_dim=8,
        num_layers=2,
        output_dim=1,
        dropout=0.0,
    )
    path = write_manual_checkpoint(
        tmp_path / "mismatch.pt",
        model_config=config,
        state_dict=big_model.state_dict(),
    )
    with pytest.raises(GNNCheckpointInvalidError):
        GNNPredictor(path)


# ---------------------------------------------------------------------------
# 2. Inference behaviour (success, count, mapping, finiteness, determinism)
# ---------------------------------------------------------------------------


def test_successful_inference(predictor: GNNPredictor) -> None:
    result = predictor.predict(make_dataset(8))
    assert isinstance(result, GNNPredictionResult)
    assert isinstance(result.node_ids, tuple)
    assert isinstance(result.predictions, tuple)
    assert all(isinstance(value, float) for value in result.predictions)
    assert result.checkpoint_epoch == predictor.epoch


@pytest.mark.parametrize("num_nodes", [1, 2, 5, 12])
def test_prediction_count_matches_node_count(
    predictor: GNNPredictor, num_nodes: int
) -> None:
    """No graph-level pooling: N nodes in, N predictions out."""
    result = predictor.predict(make_dataset(num_nodes))
    assert len(result.predictions) == num_nodes
    assert len(result.node_ids) == num_nodes


def test_node_ids_preserve_dataset_order(predictor: GNNPredictor) -> None:
    dataset = make_dataset(8)
    result = predictor.predict(dataset)
    assert result.node_ids == dataset.node_ids


def test_predictions_match_manual_forward(predictor: GNNPredictor) -> None:
    """Positional mapping: predictions[i] belongs to node_ids[i]."""
    dataset = make_dataset(8)
    result = predictor.predict(dataset)
    data = dataset.to_pyg_data()
    with torch.no_grad():
        expected = predictor.model(data.x, data.edge_index).tolist()
    assert result.predictions == pytest.approx(expected, abs=1e-6)


def test_predictions_are_finite(predictor: GNNPredictor) -> None:
    result = predictor.predict(make_dataset(8))
    values = np.asarray(result.predictions, dtype=np.float64)
    assert bool(np.isfinite(values).all())


def test_inference_is_deterministic(trained: Path) -> None:
    """Same checkpoint + same dataset => identical predictions (eval mode)."""
    dataset = make_dataset(8)
    first = GNNPredictor(trained, device="cpu").predict(dataset)
    second = GNNPredictor(trained, device="cpu").predict(dataset)
    assert first.predictions == second.predictions


def test_inference_runs_in_eval_mode(tmp_path: Path) -> None:
    """Dropout stays inactive: repeated runs of the same predictor agree."""
    path, _, dataset = train_and_save(tmp_path, dropout=0.5, filename="drop.pt")
    predictor = GNNPredictor(path, device="cpu")
    first = predictor.predict(dataset)
    second = predictor.predict(dataset)
    assert predictor.model.training is False
    assert first.predictions == second.predictions


def test_inference_works_without_targets(predictor: GNNPredictor) -> None:
    """No target leakage: inference never reads y (unlabeled graphs work)."""
    result = predictor.predict(make_dataset(8, labeled=False))
    assert len(result.predictions) == 8


def test_round_trip_matches_trainer_predict(tmp_path: Path) -> None:
    """End-to-end: a saved checkpoint reproduces the trainer's predictions."""
    path, trainer, dataset = train_and_save(tmp_path, filename="rt.pt")
    result = GNNPredictor(path, device="cpu").predict(dataset)
    expected = trainer.predict().cpu().tolist()
    assert result.predictions == pytest.approx(expected, rel=1e-6, abs=1e-8)


# ---------------------------------------------------------------------------
# 3. Input validation (invalid input is rejected loudly, never repaired)
# ---------------------------------------------------------------------------


def test_feature_width_mismatch_rejected(predictor: GNNPredictor) -> None:
    with pytest.raises(GNNModelIncompatibleError, match="input_dim"):
        predictor.predict(make_dataset(8, feature_dim=FEATURE_DIM + 1))


def test_empty_graph_rejected(predictor: GNNPredictor) -> None:
    stub = Mock(spec=GraphDataset)
    stub.num_nodes = 0
    with pytest.raises(GNNPredictionInputError, match="empty graph"):
        predictor.predict(stub)


def test_non_dataset_input_rejected(predictor: GNNPredictor) -> None:
    with pytest.raises(GNNPredictionInputError, match="GraphDataset"):
        predictor.predict("not a dataset")


def test_nan_features_rejected(predictor: GNNPredictor) -> None:
    """A real GraphDataset rejects NaN at construction; this stub proves the
    predictor's own defense-in-depth check fires before any forward pass."""
    stub = Mock(spec=GraphDataset)
    stub.num_nodes = 4
    stub.num_features = FEATURE_DIM
    stub.x = np.full((4, FEATURE_DIM), np.nan, dtype=np.float32)
    with pytest.raises(GNNPredictionInputError, match="NaN"):
        predictor.predict(stub)


def test_invalid_device_string_rejected(trained: Path) -> None:
    with pytest.raises(GNNPredictionInputError, match="not a valid torch device"):
        GNNPredictor(trained, device="warp-drive")


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="CUDA is available on this machine"
)
def test_cuda_device_rejected_when_unavailable(trained: Path) -> None:
    with pytest.raises(GNNPredictionInputError, match="CUDA"):
        GNNPredictor(trained, device="cuda")


def test_explicit_cpu_device_works(trained: Path) -> None:
    predictor = GNNPredictor(trained, device="cpu")
    assert str(predictor.device) == "cpu"
    result = predictor.predict(make_dataset(8))
    assert len(result.predictions) == 8


def test_result_length_mismatch_rejected() -> None:
    """The result contract enforces node-id/prediction cardinality itself."""
    with pytest.raises(GNNPredictionRuntimeError, match="prediction count"):
        GNNPredictionResult(
            node_ids=("n0",),
            predictions=(1.0, 2.0),
            checkpoint_epoch=1,
        )