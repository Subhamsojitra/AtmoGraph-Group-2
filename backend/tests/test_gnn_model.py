"""Unit tests for the GNN model architecture (Module 12).

These tests exercise ``app.ml.model`` - the graph neural network that Module
13 will train to predict downstream delays (node-level regression). They use
small, deterministic SYNTHETIC tensors; no Neo4j, no production data, no
training, and no accuracy claims. The model is verified only to be a
well-formed message-passing architecture that produces one finite
regression prediction per graph node.

Conventions follow ``tests/test_gnn_dataset.py`` (Module 11).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from app.ml.dataset import GraphDataset
from app.ml.exceptions import (
    GNNModelConfigError,
    GNNModelError,
    GNNModelInputError,
)
from app.ml.model import GNNConfig, GNNModel


# ---------------------------------------------------------------------------
# [TEST/SYNTHETIC] Deterministic fixtures - every value here is invented.
# ---------------------------------------------------------------------------

FEATURE_DIM = 5


def make_x(num_nodes: int, feature_dim: int = FEATURE_DIM) -> torch.Tensor:
    """Deterministic feature rows that DIFFER per node (row i encodes i).

    Distinct rows are essential: with identical features any message passing
    would aggregate identical values, so changed edges could not influence
    the output.
    """
    rows = torch.arange(num_nodes, dtype=torch.float32).unsqueeze(1)
    multipliers = torch.arange(1, feature_dim + 1, dtype=torch.float32)
    return (rows + 1.0) * multipliers / float(feature_dim * (num_nodes + 1))


def chain_edge_index(num_nodes: int) -> torch.Tensor:
    """0 -> 1 -> ... -> N-1 (upstream flows downstream, Module 11 direction)."""
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    sources = torch.arange(num_nodes - 1, dtype=torch.long)
    targets = sources + 1
    return torch.stack([sources, targets], dim=0)


def star_edge_index(num_nodes: int) -> torch.Tensor:
    """0 -> every other node (same node/edge count as the chain for N >= 2)."""
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    sources = torch.zeros(num_nodes - 1, dtype=torch.long)
    targets = torch.arange(1, num_nodes, dtype=torch.long)
    return torch.stack([sources, targets], dim=0)


@pytest.fixture()
def model() -> GNNModel:
    """A small deterministic UNTRAINED model (dropout off, eval mode)."""
    torch.manual_seed(42)
    m = GNNModel(input_dim=FEATURE_DIM, hidden_dim=16, num_layers=2, dropout=0.0)
    m.eval()
    return m


@pytest.fixture()
def sample_data() -> tuple[torch.Tensor, torch.Tensor]:
    return make_x(5), chain_edge_index(5)


# ---------------------------------------------------------------------------
# 1. Import & construction
# ---------------------------------------------------------------------------


def test_model_module_imports() -> None:
    from app.ml import GNNModel as PackageGNNModel
    from app.ml.model import GNNModel as ModuleGNNModel

    assert PackageGNNModel is ModuleGNNModel


def test_default_construction_stores_validated_config() -> None:
    torch.manual_seed(0)
    m = GNNModel(input_dim=5)
    assert isinstance(m, torch.nn.Module)
    assert m.config == GNNConfig(input_dim=5)
    assert m.config.hidden_dim == 64
    assert m.config.num_layers == 2
    assert m.config.output_dim == 1


def test_from_config_matches_keyword_construction() -> None:
    config = GNNConfig(
        input_dim=7, hidden_dim=9, num_layers=3, output_dim=1, dropout=0.25
    )
    torch.manual_seed(0)
    via_config = GNNModel.from_config(config)
    torch.manual_seed(0)
    via_kwargs = GNNModel(
        input_dim=7, hidden_dim=9, num_layers=3, output_dim=1, dropout=0.25
    )
    assert via_config.config == via_kwargs.config == config


def test_model_has_trainable_parameters() -> None:
    torch.manual_seed(0)
    m = GNNModel(input_dim=4, hidden_dim=8, num_layers=2)
    named_params = list(m.named_parameters())
    assert named_params, "an untrained GNN must expose learnable parameters"
    assert all(p.requires_grad for _, p in named_params)
    names = [name for name, _ in named_params]
    assert any("convs" in name for name in names)
    assert any("regression_head" in name for name in names)
    conv_params = [name for name in names if name.startswith("convs")]
    assert len(conv_params) == 4  # weight + bias per GCNConv (2 layers)


# ---------------------------------------------------------------------------
# 2. Configuration validation (fail loudly on invalid architecture)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_input_dim", [0, -3, True, 5.0, "5", None])
def test_invalid_input_dim_rejected(bad_input_dim: object) -> None:
    with pytest.raises(GNNModelConfigError, match="input_dim"):
        GNNModel(input_dim=bad_input_dim)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_hidden", [0, -1, True])
def test_invalid_hidden_dim_rejected(bad_hidden: object) -> None:
    with pytest.raises(GNNModelConfigError, match="hidden_dim"):
        GNNModel(input_dim=4, hidden_dim=bad_hidden)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_layers", [0, -2])
def test_invalid_num_layers_rejected(bad_layers: object) -> None:
    with pytest.raises(GNNModelConfigError, match="num_layers"):
        GNNModel(input_dim=4, num_layers=bad_layers)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_output", [0, -1])
def test_invalid_output_dim_rejected(bad_output: object) -> None:
    with pytest.raises(GNNModelConfigError, match="output_dim"):
        GNNModel(input_dim=4, output_dim=bad_output)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_dropout", [-0.1, 1.0, 1.5, float("nan"), True, "0.5"]
)
def test_invalid_dropout_rejected(bad_dropout: object) -> None:
    with pytest.raises(GNNModelConfigError, match="dropout"):
        GNNModel(input_dim=4, dropout=bad_dropout)  # type: ignore[arg-type]


def test_config_dataclass_validates_directly() -> None:
    with pytest.raises(GNNModelConfigError, match="num_layers"):
        GNNConfig(input_dim=2, num_layers=0)


def test_gnn_model_errors_form_hierarchy() -> None:
    assert issubclass(GNNModelConfigError, GNNModelError)
    assert issubclass(GNNModelInputError, GNNModelError)


# ---------------------------------------------------------------------------
# 3. Forward pass: node-level regression outputs (one prediction per node)
# ---------------------------------------------------------------------------


def test_forward_returns_tensor_of_node_predictions(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, edge_index = sample_data
    with torch.no_grad():
        out = model(x, edge_index)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32


def test_output_shape_matches_module11_target_layout(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """output_dim=1 yields ``[num_nodes]`` - Module 11's ``y`` layout."""
    x, edge_index = sample_data
    with torch.no_grad():
        out = model(x, edge_index)
    assert out.shape == (5,)


@pytest.mark.parametrize("num_nodes", [1, 2, 5, 12])
def test_one_prediction_per_node(model: GNNModel, num_nodes: int) -> None:
    """No graph-level pooling: N nodes in, N predictions out."""
    with torch.no_grad():
        out = model(make_x(num_nodes), chain_edge_index(num_nodes))
    assert out.shape[0] == num_nodes, "one prediction per node is mandatory"


def test_multi_output_keeps_node_dimension() -> None:
    torch.manual_seed(3)
    m = GNNModel(
        input_dim=FEATURE_DIM, hidden_dim=8, num_layers=2, output_dim=2,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        out = m(make_x(7), chain_edge_index(7))
    assert out.shape == (7, 2)


def test_predictions_are_finite(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, edge_index = sample_data
    with torch.no_grad():
        out = model(x, edge_index)
    assert bool(torch.isfinite(out).all())


def test_single_node_with_no_edges(model: GNNModel) -> None:
    with torch.no_grad():
        out = model(make_x(1), torch.zeros((2, 0), dtype=torch.long))
    assert out.shape == (1,)
    assert bool(torch.isfinite(out).all())


def test_self_loop_edge_is_accepted(model: GNNModel) -> None:
    """Module 11 preserves Neo4j self-loops; the model must tolerate them."""
    x = make_x(3)
    edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    with torch.no_grad():
        out = model(x, edge_index)
    assert out.shape == (3,)
    assert bool(torch.isfinite(out).all())


@pytest.mark.parametrize("feature_dim", [1, 5, 17])
def test_forward_supports_various_feature_dimensions(feature_dim: int) -> None:
    torch.manual_seed(11)
    m = GNNModel(
        input_dim=feature_dim, hidden_dim=6, num_layers=2, dropout=0.0
    ).eval()
    with torch.no_grad():
        out = m(make_x(6, feature_dim), chain_edge_index(6))
    assert out.shape == (6,)
    assert bool(torch.isfinite(out).all())


# ---------------------------------------------------------------------------
# 4. Graph connectivity is actually consumed (message passing, not an MLP)
# ---------------------------------------------------------------------------


def test_edges_change_node_predictions(model: GNNModel) -> None:
    x = make_x(5)
    no_edges = torch.zeros((2, 0), dtype=torch.long)
    with torch.no_grad():
        without_edges = model(x, no_edges)
        with_edges = model(x, chain_edge_index(5))
    assert not torch.allclose(without_edges, with_edges), (
        "identical node features must produce different predictions when "
        "graph connectivity changes; otherwise edges are ignored"
    )


def test_edge_direction_is_consumed(model: GNNModel) -> None:
    """Module 11 direction (upstream -> downstream) must matter."""
    x = make_x(4)
    forward_chain = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    reversed_chain = forward_chain.flip(0)  # 1->0, 2->1, 3->2
    with torch.no_grad():
        out_forward = model(x, forward_chain)
        out_reversed = model(x, reversed_chain)
    assert not torch.allclose(out_forward, out_reversed), (
        "reversing every edge must change the predictions; the model "
        "consumes the directed Module 11 representation"
    )


def test_different_structures_produce_different_outputs(model: GNNModel) -> None:
    x = make_x(5)
    with torch.no_grad():
        chain_out = model(x, chain_edge_index(5))
        star_out = model(x, star_edge_index(5))
    assert not torch.allclose(chain_out, star_out), (
        "a chain and a star over the same nodes must not collapse to the "
        "same predictions"
    )


# ---------------------------------------------------------------------------
# 5. Determinism (seeded init; eval mode has no stochastic behavior)
# ---------------------------------------------------------------------------


def test_seeded_construction_is_reproducible() -> None:
    torch.manual_seed(123)
    a = GNNModel(input_dim=4, hidden_dim=8, num_layers=2, dropout=0.0).eval()
    torch.manual_seed(123)
    b = GNNModel(input_dim=4, hidden_dim=8, num_layers=2, dropout=0.0).eval()
    x = make_x(4, 4)
    with torch.no_grad():
        assert torch.equal(
            a(x, chain_edge_index(4)), b(x, chain_edge_index(4))
        )


def test_eval_forward_is_deterministic(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, edge_index = sample_data
    with torch.no_grad():
        first = model(x, edge_index)
        second = model(x, edge_index)
    assert torch.equal(first, second)


def test_dropout_only_active_in_train_mode() -> None:
    torch.manual_seed(5)
    m = GNNModel(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=3, dropout=0.5)
    x = make_x(5)
    edges = chain_edge_index(5)
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(x, edges), m(x, edges))
    m.train()
    with torch.no_grad():
        first = m(x, edges)
        second = m(x, edges)
    assert not torch.equal(first, second), (
        "dropout must regularize training only; eval mode stays deterministic"
    )


# ---------------------------------------------------------------------------
# 6. Forward-pass input validation (fail loudly on contract violations)
# ---------------------------------------------------------------------------


def test_forward_rejects_wrong_feature_width(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, edge_index = sample_data
    with pytest.raises(GNNModelInputError, match="input_dim"):
        model(x[:, :3], edge_index)


def test_forward_rejects_non_2d_features(model: GNNModel) -> None:
    with pytest.raises(GNNModelInputError, match="2-D"):
        model(torch.ones(3), chain_edge_index(3))


def test_forward_rejects_empty_node_set(model: GNNModel) -> None:
    with pytest.raises(GNNModelInputError, match="0 nodes"):
        model(
            torch.zeros((0, FEATURE_DIM)),
            torch.zeros((2, 0), dtype=torch.long),
        )


def test_forward_rejects_non_float_features(model: GNNModel) -> None:
    with pytest.raises(GNNModelInputError, match="float"):
        model(
            torch.ones((3, FEATURE_DIM), dtype=torch.long),
            chain_edge_index(3),
        )


def test_forward_rejects_nan_features(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, edge_index = sample_data
    x = x.clone()
    x[0, 0] = float("nan")
    with pytest.raises(GNNModelInputError, match="NaN/infinite"):
        model(x, edge_index)


def test_forward_rejects_non_long_edge_index(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, edge_index = sample_data
    with pytest.raises(GNNModelInputError, match="int64"):
        model(x, edge_index.to(torch.int32))


def test_forward_rejects_malformed_edge_index_shape(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, edge_index = sample_data
    with pytest.raises(GNNModelInputError, match=r"\[2, num_edges\]"):
        model(x, edge_index[:1])  # [1, E]
    with pytest.raises(GNNModelInputError, match=r"\[2, num_edges\]"):
        model(x, edge_index.reshape(-1))  # 1-D


def test_forward_rejects_out_of_bounds_edge_index(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, _ = sample_data
    bad = torch.tensor([[0, 5], [1, 2]], dtype=torch.long)
    with pytest.raises(GNNModelInputError, match="out of bounds"):
        model(x, bad)


def test_forward_rejects_non_tensor_inputs(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    _, edge_index = sample_data
    with pytest.raises(GNNModelInputError, match="torch.Tensor"):
        model([[0.0] * FEATURE_DIM] * 3, edge_index)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. Device support (CPU mandatory, CUDA optional)
# ---------------------------------------------------------------------------


def test_cpu_forward(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, edge_index = sample_data
    cpu_model = model.to("cpu")
    with torch.no_grad():
        out = cpu_model(x.to("cpu"), edge_index.to("cpu"))
    assert out.device.type == "cpu"
    assert out.shape == (5,)
    assert bool(torch.isfinite(out).all())


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available; CPU-only execution is fully supported",
)
def test_cuda_forward_when_available(
    model: GNNModel, sample_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, edge_index = sample_data
    cuda_model = model.to("cuda")
    with torch.no_grad():
        out = cuda_model(x.to("cuda"), edge_index.to("cuda"))
    assert out.device.type == "cuda"


# ---------------------------------------------------------------------------
# 8. Minimal state persistence (state_dict based, for Module 13 training)
# ---------------------------------------------------------------------------


def test_save_and_load_state_round_trip(
    model: GNNModel,
    sample_data: tuple[torch.Tensor, torch.Tensor],
    tmp_path,
) -> None:
    x, edge_index = sample_data
    path = model.save_state(tmp_path / "m12_state.pt")
    assert path.is_file()
    loaded = GNNModel.load_state(path)
    assert loaded.config == model.config
    loaded.eval()
    with torch.no_grad():
        assert torch.equal(loaded(x, edge_index), model(x, edge_index))


def test_load_state_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="state file"):
        GNNModel.load_state(tmp_path / "missing_state.pt")


# ---------------------------------------------------------------------------
# 9. Module 11 integration + anti-leakage guards (smoke test)
# ---------------------------------------------------------------------------


def _synthetic_dataset() -> GraphDataset:
    """A labeled Module 11 GraphDataset built from [TEST/SYNTHETIC] values."""
    x = make_x(4).numpy().astype(np.float32)
    edge_index = chain_edge_index(4).numpy().astype(np.int64)
    y = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)  # synthetic test labels
    return GraphDataset(
        node_ids=("n0", "n1", "n2", "n3"),
        x=x,
        edge_index=edge_index,
        edge_rel_types=("SUPPLIES",) * edge_index.shape[1],
        y=y,
        target_name="downstream_delay_days",  # documented future target name
        metadata=None,
    )


def test_module11_dataset_feeds_model_end_to_end() -> None:
    """GraphDataset -> to_pyg_data() -> GNNModel -> one finite prediction/node."""
    dataset = _synthetic_dataset()
    data = dataset.to_pyg_data()
    torch.manual_seed(42)
    model = GNNModel(input_dim=dataset.num_features)
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
    assert predictions.shape[0] == dataset.num_nodes
    assert bool(torch.isfinite(predictions).all())


def test_forward_signature_has_no_target_parameter() -> None:
    """Anti-leakage guard: the regression target can never be a model input."""
    params = inspect.signature(GNNModel.forward).parameters
    assert list(params) == ["self", "x", "edge_index"]
    for forbidden in ("y", "target", "target_delay", "future_delay"):
        assert forbidden not in params
