"""GNN model architecture for node-level delay regression (Module 12).

Consumes the GNN-ready :class:`~app.ml.dataset.GraphDataset` produced by
Module 11 and maps it to one numerical prediction per graph node::

    node features x  [num_nodes, input_dim]      (Module 11 encoder)
            |
    GCNConv(input_dim -> hidden_dim) + ReLU + Dropout
            |
            |  (repeated ``num_layers`` times; message passing along the
            |   Module 11 edge direction, upstream -> downstream)
            v
    node embeddings  [num_nodes, hidden_dim]
            |
    Linear regression head (hidden_dim -> output_dim)
            |
    predicted downstream delay per node  [num_nodes]   (output_dim == 1)

Why GCN (Kipf & Welling 2017)
-----------------------------
* Simple, well-supported and adequate for a first node-regression model;
  it runs on CPU with the pure-Python ``torch_geometric`` core (no compiled
  ``torch-scatter``/``torch-sparse`` companion wheels needed).
* Message passing flows source -> target along ``edge_index`` exactly as
  stored by Module 11 (``(source)-[rel]->(target)`` = "target consumes from
  source"), so upstream disruption features propagate toward downstream
  nodes — precisely the supply-chain ripple-effect task. Edges are never
  reversed here; Neo4j relationship semantics are untouched.
* GCNConv adds self-loops internally, so isolated nodes keep their own
  features instead of receiving an all-zero aggregate.

Contract (enforced, not optional)
---------------------------------
* NODE-LEVEL regression: the output has exactly one prediction per node.
  There is no pooling/readout that would collapse nodes into a graph-level
  scalar.
* NO target leakage: ``forward(x, edge_index)`` accepts only node features
  and edges. The regression target ``y`` (Module 11) is reserved for the
  Module 13 training loss and is never a model input.
* Output shape ``[num_nodes]`` when ``output_dim == 1`` — matching the
  Module 11 target layout ``y: float32 [num_nodes]`` — otherwise
  ``[num_nodes, output_dim]``.

STATUS — UNTRAINED
------------------
Weights are randomly initialized at construction. This module provides the
ARCHITECTURE only: no optimizer, no loss, no training loop, no evaluation
and no serving API (Modules 13+). Nothing here claims predictive accuracy.

Persistence uses ``torch.save`` of a plain ``{config, state_dict}`` mapping
(state-dict based, no arbitrary Python objects).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv

from app.core.logger import get_logger
from app.ml.exceptions import GNNModelConfigError, GNNModelInputError

logger = get_logger(__name__)

__all__ = ["GNNConfig", "GNNModel"]

#: Top-level keys of a serialized model state file (see ``GNNModel.save_state``).
_STATE_KEY_CONFIG = "config"
_STATE_KEY_WEIGHTS = "state_dict"

#: Accepted device specifiers for :meth:`GNNModel.load_state`.
_DeviceLike = Union[str, torch.device]


@dataclass(frozen=True)
class GNNConfig:
    """Validated hyperparameters of the GNN architecture.

    Attributes:
        input_dim: Number of node input features (from Module 11:
            ``GraphDataset.num_features``). Must be a positive integer.
        hidden_dim: Width of every GNN layer / node embedding. Positive.
        num_layers: Number of GNN message-passing layers (>= 1). One layer
            propagates information to direct neighbors; ``num_layers``
            layers reach up to ``num_layers`` hops.
        output_dim: Regression output size per node. ``1`` (the default)
            yields the per-node scalar downstream delay matching Module 11's
            ``y`` layout; values >= 2 would model multi-target regression.
        dropout: Dropout probability applied between GNN layers, in
            ``[0.0, 1.0)``. ``0.0`` disables dropout (deterministic eval).
    """

    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        _validate_positive_int("input_dim", self.input_dim, minimum=1)
        _validate_positive_int("hidden_dim", self.hidden_dim, minimum=1)
        _validate_positive_int("num_layers", self.num_layers, minimum=1)
        _validate_positive_int("output_dim", self.output_dim, minimum=1)
        _validate_dropout(self.dropout)

    def to_dict(self) -> dict[str, Any]:
        """Plain primitive dict (safe for checkpoints and logging)."""
        return asdict(self)


def _validate_positive_int(name: str, value: Any, *, minimum: int) -> int:
    """Reject non-integers (including ``bool``) and out-of-domain values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise GNNModelConfigError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    if value < minimum:
        raise GNNModelConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _validate_dropout(value: Any) -> float:
    """Dropout must be a probability in ``[0.0, 1.0)``; NaN rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GNNModelConfigError(
            f"dropout must be a float, got {type(value).__name__}"
        )
    probability = float(value)
    if math.isnan(probability) or probability < 0.0 or probability >= 1.0:
        raise GNNModelConfigError(f"dropout must be in [0.0, 1.0), got {value}")
    return probability


class GNNModel(nn.Module):
    """Graph convolutional network for node-level delay regression.

    The model is a stack of :class:`~torch_geometric.nn.GCNConv` layers
    followed by a linear regression head. It performs real graph message
    passing (never a plain MLP): every layer aggregates neighbor features
    along ``edge_index`` before the head maps node embeddings to delays.

    Example:
        >>> model = GNNModel(input_dim=5, hidden_dim=32, num_layers=2)
        >>> predictions = model(x, edge_index)   # shape [num_nodes]

    Inputs are validated on every forward pass (shape, dtype, finiteness,
    index bounds) so mis-shaped data fails loudly instead of silently
    corrupting a later training run.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = GNNConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
        )

        convs: list[GCNConv] = []
        layer_input_dim = self.config.input_dim
        for _ in range(self.config.num_layers):
            convs.append(GCNConv(layer_input_dim, self.config.hidden_dim))
            layer_input_dim = self.config.hidden_dim
        self.convs = nn.ModuleList(convs)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.regression_head = nn.Linear(
            self.config.hidden_dim, self.config.output_dim
        )

        logger.info(
            "GNN model initialized (untrained architecture)",
            extra=self.config.to_dict(),
        )

    @classmethod
    def from_config(cls, config: GNNConfig) -> "GNNModel":
        """Build a model from an existing validated :class:`GNNConfig`."""
        return cls(
            config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_dim=config.output_dim,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------ #
    # Forward pass (architecture only — no training logic here)
    # ------------------------------------------------------------------ #

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Predict one downstream-delay value per node.

        Args:
            x: Float node features ``[num_nodes, input_dim]`` (Module 11
                produces float32).
            edge_index: Int64 edge list ``[2, num_edges]``; row 0 = source,
                row 1 = target, direction exactly as stored by Module 11
                (upstream -> downstream). Zero edges (empty second column)
                are valid: every node then keeps only its self-loop message.

        Returns:
            Tensor of predictions with shape ``[num_nodes]`` when
            ``output_dim == 1`` (default; matches Module 11's ``y`` layout),
            otherwise ``[num_nodes, output_dim]``. One row per node — never
            pooled into a graph-level value.
        """
        self._validate_inputs(x, edge_index)

        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.activation(h)
            h = self.dropout(h)
        out = self.regression_head(h)

        if self.config.output_dim == 1:
            return out.squeeze(-1)
        return out

    # ------------------------------------------------------------------ #
    # Input validation (fail loudly, matching Module 11 conventions)
    # ------------------------------------------------------------------ #

    def _validate_inputs(self, x: Tensor, edge_index: Tensor) -> None:
        """Raise :class:`GNNModelInputError` on any contract violation."""
        if not isinstance(x, torch.Tensor):
            raise GNNModelInputError(
                f"x must be a torch.Tensor, got {type(x).__name__}"
            )
        if x.ndim != 2:
            raise GNNModelInputError(
                "x must be 2-D [num_nodes, num_features], got shape "
                f"{tuple(x.shape)}"
            )
        if x.shape[0] == 0:
            raise GNNModelInputError(
                "x contains 0 nodes; a prediction requires at least one node"
            )
        if x.shape[1] != self.config.input_dim:
            raise GNNModelInputError(
                f"x has {x.shape[1]} feature columns but the model was built "
                f"with input_dim={self.config.input_dim}"
            )
        if not x.is_floating_point():
            raise GNNModelInputError(
                f"x must be a float tensor (Module 11 provides float32), "
                f"got dtype {x.dtype}"
            )
        if not bool(torch.isfinite(x).all()):
            raise GNNModelInputError(
                "x contains NaN/infinite values; corrupt inputs must fail "
                "loudly, never flow through the network"
            )

        if not isinstance(edge_index, torch.Tensor):
            raise GNNModelInputError(
                f"edge_index must be a torch.Tensor, got "
                f"{type(edge_index).__name__}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise GNNModelInputError(
                "edge_index must be 2-D [2, num_edges], got shape "
                f"{tuple(edge_index.shape)}"
            )
        if edge_index.dtype != torch.long:
            raise GNNModelInputError(
                f"edge_index must have dtype torch.int64 (Module 11 layout), "
                f"got {edge_index.dtype}"
            )
        if edge_index.shape[1]:
            lo = int(edge_index.min())
            hi = int(edge_index.max())
            if lo < 0 or hi >= x.shape[0]:
                raise GNNModelInputError(
                    f"edge_index out of bounds: valid indices are "
                    f"[0, {x.shape[0] - 1}], found [{lo}, {hi}]"
                )

    # ------------------------------------------------------------------ #
    # Minimal state persistence for later modules (state_dict based)
    # ------------------------------------------------------------------ #

    def save_state(self, path: Union[str, Path]) -> Path:
        """Persist architecture config + weights; returns the written path.

        The file contains a plain ``{"config": {...}, "state_dict": {...}}``
        mapping (tensors and primitives only — no pickled Python objects).
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                _STATE_KEY_CONFIG: self.config.to_dict(),
                _STATE_KEY_WEIGHTS: self.state_dict(),
            },
            target_path,
        )
        logger.info("GNN model state saved", extra={"path": str(target_path)})
        return target_path

    @classmethod
    def load_state(
        cls,
        path: Union[str, Path],
        *,
        map_location: Optional[_DeviceLike] = "cpu",
    ) -> "GNNModel":
        """Rebuild a model (config + weights) from a ``save_state`` file.

        ``map_location`` defaults to CPU so saved states load without a GPU.
        """
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"GNN model state file not found: {source_path}"
            )
        payload = torch.load(
            source_path, map_location=map_location, weights_only=True
        )
        if not isinstance(payload, Mapping):
            raise GNNModelConfigError(
                f"Unrecognized GNN state file format: {source_path}"
            )
        model = cls.from_config(GNNConfig(**payload[_STATE_KEY_CONFIG]))
        model.load_state_dict(payload[_STATE_KEY_WEIGHTS])
        logger.info("GNN model state loaded", extra={"path": str(source_path)})
        return model

