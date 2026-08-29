"""Unit tests for GNN training & evaluation (Module 13).

Fast, deterministic, CPU-only tests around ``app.ml.training``,
``app.ml.evaluation`` and ``app.ml.splitting``. Every fixture below is
explicitly [TEST/SYNTHETIC]: no Neo4j, no internet, no GPU, no downloads and
no accuracy claims. The synthetic problems prove that the pipeline TRAINS,
EVALUATES, SPLITS and CHECKPOINTS correctly - they do NOT claim real-world
predictive accuracy (the project database contains no real delay labels).

Conventions follow ``tests/test_gnn_model.py`` (Module 12).
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from app.ml.dataset import GraphDataset, GraphDatasetBuilder
from app.ml.evaluation import regression_metrics
from app.ml.exceptions import (
    GNNCheckpointError,
    GNNEvaluationError,
    GNNTrainingConfigError,
    GNNTrainingDataError,
    GNNTrainingError,
)
from app.ml.model import GNNModel
from app.ml.splitting import split_nodes
from app.ml.training import GNNTrainingConfig, GNNTrainer, set_seed
from app.repositories.graph_repository import GraphRepository


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
    y: np.ndarray | None = None,
    edge_index: np.ndarray | None = None,
) -> GraphDataset:
    """Small labeled (or unlabeled) Module 11 GraphDataset from synthetic values."""
    if y is None and labeled:
        y = np.linspace(0.1, 0.1 + 0.1 * (num_nodes - 1), num_nodes).astype(
            np.float32
        )
    edge_index = chain_edge_index(num_nodes) if edge_index is None else edge_index
    return GraphDataset(
        node_ids=tuple(f"n{i}" for i in range(num_nodes)),
        x=make_x(num_nodes),
        edge_index=edge_index,
        edge_rel_types=("SUPPLIES",) * edge_index.shape[1],
        y=y,
        target_name="downstream_delay_days" if labeled else None,
        metadata=None,
    )


def make_model(*, hidden_dim: int = 8, dropout: float = 0.0) -> GNNModel:
    """Small deterministic UNTRAINED Module 12 model."""
    torch.manual_seed(42)
    return GNNModel(
        input_dim=FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_layers=2,
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


# ---------------------------------------------------------------------------
# 1. Training configuration validation (Step 25)
# ---------------------------------------------------------------------------


def test_default_config_is_valid() -> None:
    config = GNNTrainingConfig()
    assert config.epochs == 200
    assert config.learning_rate == 0.01
    assert config.weight_decay == 5e-4
    assert config.device == "cpu"
    assert config.loss == "mse"
    assert config.seed is None
    assert config.patience is None
    assert config.restore_best is True
    assert config.eval_interval == 1
    assert config.checkpoint_path is None


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_epochs_must_be_positive_int(bad: object) -> None:
    with pytest.raises(GNNTrainingConfigError, match="epochs"):
        make_config(epochs=bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf"), "fast"])
def test_learning_rate_must_be_positive_finite(bad: object) -> None:
    with pytest.raises(GNNTrainingConfigError, match="learning_rate"):
        make_config(learning_rate=bad)


@pytest.mark.parametrize("bad", [-1e-9, float("nan")])
def test_weight_decay_must_be_non_negative(bad: object) -> None:
    with pytest.raises(GNNTrainingConfigError, match="weight_decay"):
        make_config(weight_decay=bad)


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_seed_must_be_non_negative_int_or_none(bad: object) -> None:
    with pytest.raises(GNNTrainingConfigError, match="seed"):
        make_config(seed=bad)


def test_loss_name_is_normalized_and_validated() -> None:
    assert make_config(loss=" MAE ").loss == "mae"
    with pytest.raises(GNNTrainingConfigError, match="unknown loss"):
        make_config(loss="bce")


@pytest.mark.parametrize("bad", [1.0, -0.1, float("nan"), 0.0])
def test_validation_split_must_be_in_open_interval(bad: object) -> None:
    with pytest.raises(GNNTrainingConfigError, match="validation_split"):
        make_config(validation_split=bad)


def test_split_ratios_must_sum_below_one() -> None:
    with pytest.raises(GNNTrainingConfigError, match="must be < 1.0"):
        make_config(validation_split=0.6, test_split=0.6)


def test_patience_and_eval_interval_are_validated() -> None:
    assert make_config(patience=0).patience == 0
    with pytest.raises(GNNTrainingConfigError, match="patience"):
        make_config(patience=-1)
    with pytest.raises(GNNTrainingConfigError, match="eval_interval"):
        make_config(eval_interval=0)


def test_checkpoint_path_is_normalized_to_path() -> None:
    config = make_config(checkpoint_path="checkpoints/m13.pt")
    assert isinstance(config.checkpoint_path, Path)
    with pytest.raises(GNNTrainingConfigError, match="checkpoint_path"):
        make_config(checkpoint_path=123)  # type: ignore[arg-type]


def test_invalid_device_string_rejected_by_config() -> None:
    with pytest.raises(GNNTrainingConfigError, match="valid torch device"):
        make_config(device="not-a-device")


# ---------------------------------------------------------------------------
# 2. Regression metrics (Step 13 / Step 22)
# ---------------------------------------------------------------------------


def test_metric_names_and_exact_values() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert set(metrics) == {"mse", "rmse", "mae", "r2"}
    assert metrics["mse"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt(1.0 / 3.0))
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["r2"] == pytest.approx(11.0 / 14.0)


def test_perfect_predictions_score_r2_one() -> None:
    target = [0.5, 1.5, 2.5]
    metrics = regression_metrics(target, target)
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == pytest.approx(1.0)


def test_constant_target_r2_is_none_but_other_metrics_defined() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert metrics["r2"] is None  # zero target variance -> undefined, not NaN
    assert metrics["mse"] == pytest.approx(2.0 / 3.0)
    assert metrics["mae"] == pytest.approx(2.0 / 3.0)


def test_single_sample_r2_is_undefined() -> None:
    metrics = regression_metrics([1.5], [1.0])
    assert metrics["r2"] is None
    assert metrics["mse"] == pytest.approx(0.25)


def test_accepts_torch_tensors_numpy_and_lists() -> None:
    reference = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    from_tensor = regression_metrics(
        torch.tensor([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])
    )
    assert from_tensor == reference


def test_empty_inputs_rejected() -> None:
    with pytest.raises(GNNEvaluationError, match="0 nodes"):
        regression_metrics([], [])


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(GNNEvaluationError, match="same shape"):
        regression_metrics([1.0, 2.0], [1.0, 2.0, 3.0])


def test_non_1d_inputs_rejected() -> None:
    with pytest.raises(GNNEvaluationError, match="1-D"):
        regression_metrics([[1.0, 2.0]], [[1.0, 2.0]])


def test_nan_and_infinite_values_rejected() -> None:
    with pytest.raises(GNNEvaluationError, match="NaN/infinite"):
        regression_metrics([1.0, float("nan")], [1.0, 2.0])
    with pytest.raises(GNNEvaluationError, match="NaN/infinite"):
        regression_metrics([1.0, float("inf")], [1.0, 2.0])
    with pytest.raises(GNNEvaluationError, match="NaN/infinite"):
        regression_metrics([1.0, 2.0], [1.0, float("-inf")])


# ---------------------------------------------------------------------------
# 3. Deterministic node splitting (Step 7 / Step 23)
# ---------------------------------------------------------------------------


def test_split_is_deterministic_with_seed() -> None:
    first = split_nodes(50, 0.2, 0.1, seed=123)
    second = split_nodes(50, 0.2, 0.1, seed=123)
    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.val_indices, second.val_indices)
    assert np.array_equal(first.test_indices, second.test_indices)


def test_split_counts_no_overlap_and_full_coverage() -> None:
    split = split_nodes(10, 0.2, 0.1, seed=0)
    assert split.sizes == (7, 2, 1)
    combined = np.concatenate(
        [split.train_indices, split.val_indices, split.test_indices]
    )
    assert sorted(combined.tolist()) == list(range(10))  # disjoint + complete


def test_split_scales_with_ratio() -> None:
    assert split_nodes(100, 0.2, 0.1, seed=1).sizes == (70, 20, 10)


def test_zero_test_split_keeps_all_nodes_in_train_val() -> None:
    split = split_nodes(10, 0.5, 0.0, seed=2)
    assert split.sizes == (5, 5, 0)


@pytest.mark.parametrize("val,test", [(1.0, 0.0), (-0.1, 0.0), (0.5, 0.6), (0.2, 0.9)])
def test_invalid_split_ratios_rejected(val: float, test: float) -> None:
    with pytest.raises(GNNTrainingConfigError):
        split_nodes(10, val, test, seed=0)


def test_empty_dataset_rejected() -> None:
    with pytest.raises(GNNTrainingDataError, match="cannot split"):
        split_nodes(0, 0.2, 0.1, seed=0)


def test_dataset_too_small_for_requested_split_rejected() -> None:
    with pytest.raises(GNNTrainingDataError, match="too small"):
        split_nodes(1, 0.2, 0.0, seed=0)
    with pytest.raises(GNNTrainingDataError, match="too small"):
        split_nodes(2, 0.5, 0.4, seed=0)


def test_non_integer_num_nodes_rejected() -> None:
    with pytest.raises(GNNTrainingDataError, match="integer"):
        split_nodes("10", 0.2, 0.1, seed=0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. Trainer construction & target validation (Step 6 / Step 18)
# ---------------------------------------------------------------------------


def test_trainer_rejects_unlabeled_dataset() -> None:
    with pytest.raises(GNNTrainingDataError, match="no targets"):
        GNNTrainer(make_model(), make_dataset(4, labeled=False), make_config())


def test_trainer_rejects_empty_graph() -> None:
    """Defense-in-depth guard: Module 11 already refuses to CONSTRUCT a
    0-node dataset (``EmptyGraphError``), so the trainer guard is exercised
    directly via its validation hook."""
    empty = Mock(spec=GraphDataset)
    empty.num_nodes = 0
    with pytest.raises(GNNTrainingDataError, match="empty graph"):
        GNNTrainer._validate_dataset_for_training(empty, make_model())


def test_trainer_rejects_multi_output_model() -> None:
    torch.manual_seed(0)
    model = GNNModel(input_dim=FEATURE_DIM, output_dim=2)
    with pytest.raises(GNNTrainingConfigError, match="output_dim"):
        GNNTrainer(model, make_dataset(4), make_config())


def test_trainer_rejects_non_gnn_model_and_non_dataset() -> None:
    with pytest.raises(GNNTrainingError, match="GNNModel"):
        GNNTrainer(torch.nn.Linear(2, 2), make_dataset(4), make_config())
    with pytest.raises(GNNTrainingError, match="GraphDataset"):
        GNNTrainer(make_model(), {"x": 1}, make_config())  # type: ignore[arg-type]


def test_trainer_uses_cpu_and_moves_model_in_place() -> None:
    model = make_model()
    trainer = GNNTrainer(model, make_dataset(8), make_config())
    assert trainer.device.type == "cpu"
    assert trainer.model is model
    assert trainer.split.sizes == (5, 2, 1)  # 8 nodes; round(1.6)=2 val, round(0.8)=1 test


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="CUDA is available; there is no missing-GPU error to assert",
)
def test_cuda_device_rejected_when_unavailable() -> None:
    with pytest.raises(GNNTrainingError, match="CUDA"):
        GNNTrainer(make_model(), make_dataset(4), make_config(device="cuda"))


# ---------------------------------------------------------------------------
# 5. Training smoke test + evaluation API (Step 12 / Step 20)
# ---------------------------------------------------------------------------


def test_training_smoke_run_completes_with_finite_history() -> None:
    dataset = make_dataset(8)
    trainer = GNNTrainer(make_model(), dataset, make_config(epochs=3, seed=7))
    history = trainer.train()

    assert history.epochs_completed == 3
    assert len(history.train_loss) == 3
    assert all(math.isfinite(value) for value in history.train_loss)
    assert all(
        value is not None and math.isfinite(value) for value in history.val_loss
    )
    assert all(
        set(entry) == {"mse", "rmse", "mae", "r2"} for entry in history.train_metrics
    )
    assert all(
        entry is not None and set(entry) == {"mse", "rmse", "mae", "r2"}
        for entry in history.val_metrics
    )


def test_prediction_count_matches_node_count() -> None:
    dataset = make_dataset(8)
    trainer = GNNTrainer(make_model(), dataset, make_config())
    predictions = trainer.predict()
    assert predictions.shape[0] == dataset.num_nodes
    assert predictions.ndim == 1
    assert bool(torch.isfinite(predictions).all())


def test_predict_subset_and_bounds() -> None:
    trainer = GNNTrainer(make_model(), make_dataset(8), make_config())
    assert trainer.predict([0, 3]).shape[0] == 2
    with pytest.raises(GNNEvaluationError, match="out of bounds"):
        trainer.predict([99])


def test_validate_and_evaluate_test_are_finite() -> None:
    trainer = GNNTrainer(make_model(), make_dataset(8), make_config())
    val_loss, val_metrics = trainer.validate()
    assert math.isfinite(val_loss)
    assert set(val_metrics) == {"mse", "rmse", "mae", "r2"}
    test_loss, test_metrics = trainer.evaluate_test()
    assert math.isfinite(test_loss)
    assert set(test_metrics) == {"mse", "rmse", "mae", "r2"}


def test_evaluate_test_without_test_split_raises() -> None:
    trainer = GNNTrainer(
        make_model(), make_dataset(8), make_config(test_split=0.0)
    )
    with pytest.raises(GNNTrainingDataError, match="no test nodes"):
        trainer.evaluate_test()


def test_evaluate_all_nodes_and_explicit_indices() -> None:
    trainer = GNNTrainer(make_model(), make_dataset(8), make_config())
    loss_all, _ = trainer.evaluate()
    loss_two, _ = trainer.evaluate([0, 1])
    assert math.isfinite(loss_all)
    assert math.isfinite(loss_two)
    with pytest.raises(GNNEvaluationError, match="out of bounds"):
        trainer.evaluate([8])


def test_history_to_dict_matches_documented_schema() -> None:
    trainer = GNNTrainer(make_model(), make_dataset(8), make_config(epochs=2))
    payload = trainer.train().to_dict()
    assert {"train_loss", "val_loss", "train_metrics", "val_metrics"} <= set(payload)
    assert payload["train_loss"] == trainer.history.train_loss  # type: ignore[union-attr]
    assert payload["best_val_loss"] == min(payload["val_loss"])


# ---------------------------------------------------------------------------
# 6. Learning progress on a solvable synthetic problem (Step 21)
# ---------------------------------------------------------------------------


def test_training_reduces_loss_on_learnable_problem() -> None:
    """An edgeless graph makes GCNConv act on self-loops only, so the linear
    rule ``y = 2*x0 + 0.5`` is exactly realizable by the network. The test
    asserts learning PROGRESS (final < initial), never an exact loss value."""
    num_nodes = 8
    x = make_x(num_nodes)
    targets = (2.0 * x[:, 0] + 0.5).astype(np.float32)
    dataset = make_dataset(
        num_nodes, y=targets, edge_index=np.zeros((2, 0), dtype=np.int64)
    )
    trainer = GNNTrainer(
        make_model(hidden_dim=16),
        dataset,
        make_config(epochs=150, learning_rate=0.05, weight_decay=0.0, seed=0),
    )
    history = trainer.train()

    assert all(math.isfinite(value) for value in history.train_loss)
    initial, final = history.train_loss[0], history.train_loss[-1]
    assert final < initial


# ---------------------------------------------------------------------------
# 7. Early stopping (Step 16)
# ---------------------------------------------------------------------------


def test_early_stopping_stops_and_tracks_best() -> None:
    num_nodes = 10
    generator = np.random.default_rng(0)
    targets = generator.uniform(0.0, 5.0, num_nodes).astype(np.float32)
    dataset = make_dataset(num_nodes, y=targets)
    trainer = GNNTrainer(
        make_model(hidden_dim=8),
        dataset,
        make_config(epochs=40, patience=1, learning_rate=0.05, seed=3),
    )
    history = trainer.train()

    assert history.stopped_early is True
    assert history.epochs_completed < 40
    evaluated = [value for value in history.val_loss if value is not None]
    assert evaluated and history.best_val_loss == min(evaluated)
    assert history.best_epoch is not None and history.best_epoch >= 1


def test_no_early_stopping_without_patience() -> None:
    trainer = GNNTrainer(make_model(), make_dataset(8), make_config(epochs=4))
    history = trainer.train()
    assert history.stopped_early is False
    assert history.epochs_completed == 4


# ---------------------------------------------------------------------------
# 8. Reproducibility (Step 15)
# ---------------------------------------------------------------------------


def test_set_seed_makes_torch_draws_reproducible() -> None:
    set_seed(123)
    first = torch.rand(4)
    set_seed(123)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_same_seed_reproduces_full_training_run() -> None:
    losses: list[list[float]] = []
    for _ in range(2):
        torch.manual_seed(11)
        model = GNNModel(
            input_dim=FEATURE_DIM, hidden_dim=8, num_layers=2, dropout=0.1
        )
        trainer = GNNTrainer(
            model,
            make_dataset(8),
            make_config(epochs=5, seed=11, learning_rate=0.02),
        )
        losses.append(trainer.train().train_loss)
    assert losses[0] == losses[1]


def test_different_seed_changes_the_training_trajectory() -> None:
    first_losses: list[float] = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        model = GNNModel(
            input_dim=FEATURE_DIM, hidden_dim=8, num_layers=2, dropout=0.0
        )
        trainer = GNNTrainer(
            model, make_dataset(8), make_config(epochs=2, seed=seed)
        )
        first_losses.append(trainer.train().train_loss[0])
    assert first_losses[0] != first_losses[1]


# ---------------------------------------------------------------------------
# 9. Checkpoint save / load (Step 17 / Step 24)
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_preserves_predictions(tmp_path) -> None:
    dataset = make_dataset(8)
    model = make_model()
    trainer = GNNTrainer(model, dataset, make_config(epochs=5, seed=5))
    history = trainer.train()
    path = trainer.save_checkpoint(tmp_path / "m13_ckpt.pt")
    assert path.is_file()

    loaded = GNNTrainer.load_checkpoint(path)
    assert loaded.epoch == 5
    assert loaded.best_val_loss == history.best_val_loss
    assert loaded.training_config == trainer.config
    assert loaded.model_config == model.config
    assert loaded.history is not None
    assert loaded.history["train_loss"] == history.train_loss

    x = torch.from_numpy(dataset.x).float()
    edge_index = torch.from_numpy(dataset.edge_index).long()
    model.eval()
    loaded.model.eval()
    with torch.no_grad():
        assert torch.allclose(
            model(x, edge_index), loaded.model(x, edge_index), atol=1e-6
        )


def test_checkpoint_load_into_existing_model(tmp_path) -> None:
    dataset = make_dataset(8)
    trainer = GNNTrainer(
        make_model(), dataset, make_config(epochs=3, seed=5)
    )
    path = trainer.save_checkpoint(tmp_path / "c.pt")

    fresh = GNNModel(
        input_dim=FEATURE_DIM, hidden_dim=8, num_layers=2, dropout=0.0
    )
    loaded = GNNTrainer.load_checkpoint(path, model=fresh)
    assert loaded.model is fresh
    fresh.eval()
    x = torch.from_numpy(dataset.x).float()
    edge_index = torch.from_numpy(dataset.edge_index).long()
    with torch.no_grad():
        assert torch.allclose(trainer.model(x, edge_index), fresh(x, edge_index))


def test_checkpoint_payload_contains_documented_keys(tmp_path) -> None:
    trainer = GNNTrainer(
        make_model(), make_dataset(8), make_config(epochs=2, seed=5)
    )
    trainer.train()
    path = trainer.save_checkpoint(tmp_path / "c.pt")
    payload = torch.load(path, weights_only=True)
    assert {
        "format_version",
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "best_val_loss",
        "training_config",
        "model_config",
        "metrics",
        "history",
    } <= set(payload)
    assert payload["epoch"] == 2
    assert payload["training_config"]["epochs"] == 2
    assert payload["model_config"]["input_dim"] == FEATURE_DIM


def test_checkpoint_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        GNNTrainer.load_checkpoint(tmp_path / "missing.pt")


def test_checkpoint_rejects_foreign_payload(tmp_path) -> None:
    path = tmp_path / "foreign.pt"
    torch.save({"unrelated": [1, 2, 3]}, path)
    with pytest.raises(GNNCheckpointError, match="missing required"):
        GNNTrainer.load_checkpoint(path)


def test_config_checkpoint_path_is_saved_after_training(tmp_path) -> None:
    target = tmp_path / "auto" / "ckpt.pt"
    trainer = GNNTrainer(
        make_model(),
        make_dataset(8),
        make_config(epochs=2, seed=5, checkpoint_path=str(target)),
    )
    trainer.train()
    assert target.is_file()


# ---------------------------------------------------------------------------
# 10. Module 11 builder integration (mocked repository - no Neo4j needed)
# ---------------------------------------------------------------------------


def _n(node_id: str, risk_score: float, delay: float) -> dict[str, object]:
    """A repository node row with a [TEST/SYNTHETIC] downstream-delay label."""
    return {
        "n": {
            "id": node_id,
            "name": node_id.replace("_", " ").title(),
            "risk_score": risk_score,
            "labels": ["Port" if node_id.endswith("A") else "Factory"],
            "downstream_delay_days": delay,
        }
    }


def _e(source: str, target: str, rel_type: str = "SUPPLIES") -> dict[str, str]:
    return {"source_id": source, "target_id": target, "rel_type": rel_type}


def make_repo(nodes: list[dict[str, object]], edges: list[dict[str, str]]) -> Mock:
    repo = Mock(spec=GraphRepository)
    repo.get_nodes.return_value = nodes
    repo.find_all_relationships.return_value = edges
    return repo


def test_module11_builder_dataset_trains_end_to_end() -> None:
    nodes = [
        _n("entity_A", 20.0, 0.5),
        _n("entity_B", 40.0, 1.0),
        _n("entity_C", 60.0, 1.5),
        _n("entity_D", 80.0, 2.0),
    ]
    edges = [
        _e("entity_A", "entity_B"),
        _e("entity_B", "entity_C"),
        _e("entity_C", "entity_D"),
    ]
    dataset = GraphDatasetBuilder(repository=make_repo(nodes, edges)).build(
        target_property="downstream_delay_days"
    )
    assert dataset.has_targets
    assert dataset.num_nodes == 4
    assert dataset.y is not None and float(dataset.y[0]) == pytest.approx(0.5)

    trainer = GNNTrainer(
        make_model(),
        dataset,
        make_config(epochs=2, seed=1, validation_split=0.25, test_split=0.25),
    )
    history = trainer.train()
    assert all(math.isfinite(value) for value in history.train_loss)
    test_loss, _ = trainer.evaluate_test()
    assert math.isfinite(test_loss)





