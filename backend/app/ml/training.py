"""GNN training and evaluation pipeline (Module 13).

Provides the reusable training/evaluation layer AROUND the existing modules —
it reuses them unchanged and contains no model or dataset code of its own::

    Module 11  GraphDataset (x, edge_index, y, node_ids)
                        |   to_pyg_data()  (tensors)
                        v
    Module 12  GNNModel(x, edge_index)  ->  [num_nodes] predictions
                        |
    Module 13  GNNTrainer (this module)
                        |   node-level regression loss on TRAIN nodes only
                        v
    Validation / Evaluation  (val/test node subsets, eval mode, no grads)
                        |
                        v
    TrainingHistory + metrics  +  state_dict checkpoints (GNNCheckpoint)

Training objective
------------------
Node-level regression: for N graph nodes the model produces N predictions and
the loss compares them against ground-truth downstream-delay targets ONLY on
the training nodes. The target ``y`` is never a model input
(``GNNModel.forward(x, edge_index)`` — anti-leakage by construction).

Design decisions (documented, deliberate)
-----------------------------------------
* LOSS: ``MSELoss`` by default (standard for regression, differentiable,
  well-behaved on CPU); ``mae`` and ``huber`` are configurable alternatives
  via ``GNNTrainingConfig(loss=...)``. No custom loss was invented because no
  specification requires one.
* OPTIMIZER: ``Adam`` with configurable ``learning_rate`` / ``weight_decay``
  (standard, well-supported). Defaults: ``lr=0.01``, ``weight_decay=5e-4``
  (the conventional GCN training values of Kipf & Welling 2017).
* SPLIT: transductive node masking (see ``app.ml.splitting``) — Module 11
  yields one connected graph, targets never enter features or the model, and
  validation/test targets are used only by their own loss/metrics.
* REPRODUCIBILITY: ``GNNTrainingConfig.seed`` seeds PyTorch's global RNG at
  the start of ``train()`` (dropout and any subsequent randomness draw from
  it). The node split uses its own locally-seeded numpy Generator, so the
  interpreter-global ``random``/``numpy.random`` state is never modified.
* DEVICE: CPU-first (the development environment has no CUDA). ``device`` is
  configurable; requesting CUDA on a CPU-only machine fails loudly.
* CHECKPOINTS: ``torch.save`` of a plain mapping of tensors + primitives
  (model/optimizer ``state_dict``, epoch, best validation loss, configs,
  metrics, history) and ``torch.load(..., weights_only=True)`` on the way
  back — no arbitrary object deserialization, same policy as Module 12.

STATUS — OFFLINE TRAINING ONLY: no HTTP endpoint, no model serving and no
real-time inference belong to this module (later modules consume
``GNNTrainer.model`` / checkpoints). No accuracy is claimed anywhere: the
project database contains no real downstream-delay labels yet, so training is
exercised on clearly-marked synthetic test data only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn

from app.core.logger import get_logger
from app.ml.dataset import GraphDataset
from app.ml.evaluation import regression_metrics
from app.ml.exceptions import (
    GNNCheckpointError,
    GNNEvaluationError,
    GNNModelConfigError,
    GNNTrainingConfigError,
    GNNTrainingDataError,
    GNNTrainingError,
)
from app.ml.model import GNNConfig, GNNModel
from app.ml.splitting import NodeSplit, split_nodes

logger = get_logger(__name__)

__all__ = [
    "SUPPORTED_LOSSES",
    "GNNCheckpoint",
    "GNNTrainingConfig",
    "GNNTrainer",
    "TrainingHistory",
    "set_seed",
]

#: Loss factories exposed by ``GNNTrainingConfig.loss`` (name -> nn.Module).
#: ``mse`` is the documented default for node-level delay regression.
SUPPORTED_LOSSES: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
}

#: Top-level keys of a trainer checkpoint file (see ``GNNTrainer.save_checkpoint``).
_CK_FORMAT = "format_version"
_CK_MODEL_STATE = "model_state_dict"
_CK_OPTIMIZER_STATE = "optimizer_state_dict"
_CK_EPOCH = "epoch"
_CK_BEST_VAL = "best_val_loss"
_CK_TRAINING_CONFIG = "training_config"
_CK_MODEL_CONFIG = "model_config"
_CK_METRICS = "metrics"
_CK_HISTORY = "history"

_CHECKPOINT_FORMAT_VERSION = 1


def set_seed(seed: int) -> None:
    """Seed PyTorch's global RNG (deliberately narrow reproducibility util).

    Only torch's RNG is seeded — weight initialization and dropout draw from
    it. The node split uses its own locally-seeded ``numpy.random.Generator``
    (see ``app.ml.splitting``), and the interpreter-global ``random`` /
    ``numpy.random`` state is never modified by Module 13.
    """
    torch.manual_seed(int(seed))


@dataclass(frozen=True)
class GNNTrainingConfig:
    """Validated hyperparameters of a Module 13 training run.

    Attributes:
        epochs: Number of training epochs (>= 1).
        learning_rate: Adam step size (> 0).
        weight_decay: Adam L2 penalty (>= 0; ``5e-4`` is the conventional
            GCN value, ``0.0`` disables regularization).
        seed: Optional RNG seed; when set, torch's global RNG is seeded at
            the start of ``train()`` and the node split is deterministic.
        device: Torch device (``"cpu"`` default; ``"cuda"`` only honoured
            when a GPU exists — checked loudly at trainer construction).
        validation_split: Validation node fraction; must be > 0 (training
            always validates) and < 1.
        test_split: Held-out test node fraction in ``[0, 1)``; ``0.0``
            disables the test evaluation.
        patience: Optional early-stopping patience (>= 0); ``None`` disables.
        eval_interval: Validate every k-th epoch (>= 1); the final epoch is
            always validated.
        restore_best: Reload best-validation-loss weights after training.
        loss: Regression loss name, one of ``sorted(SUPPORTED_LOSSES)``
            (case-insensitive). Default ``"mse"``.
        checkpoint_path: Optional path; when set, ``train()`` automatically
            saves a checkpoint there after the run finishes.
    """

    epochs: int = 200
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    seed: Optional[int] = None
    device: str = "cpu"
    validation_split: float = 0.2
    test_split: float = 0.1
    patience: Optional[int] = None
    eval_interval: int = 1
    restore_best: bool = True
    loss: str = "mse"
    checkpoint_path: Optional[Union[str, Path]] = None

    def __post_init__(self) -> None:
        _validate_int("epochs", self.epochs, minimum=1)
        _validate_finite_float("learning_rate", self.learning_rate)
        if self.learning_rate <= 0.0:
            raise GNNTrainingConfigError(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )
        _validate_finite_float("weight_decay", self.weight_decay)
        if self.weight_decay < 0.0:
            raise GNNTrainingConfigError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )

        if self.seed is not None:
            _validate_int("seed", self.seed, minimum=0)

        if not isinstance(self.device, str) or not self.device.strip():
            raise GNNTrainingConfigError(
                f"device must be a non-empty string, got {self.device!r}"
            )
        try:
            torch.device(self.device)
        except (RuntimeError, ValueError) as exc:
            raise GNNTrainingConfigError(
                f"device '{self.device}' is not a valid torch device: {exc}"
            ) from exc

        _validate_finite_float("validation_split", self.validation_split)
        if not 0.0 < self.validation_split < 1.0:
            raise GNNTrainingConfigError(
                "validation_split must be > 0 (training always validates) "
                f"and < 1, got {self.validation_split}"
            )
        _validate_finite_float("test_split", self.test_split)
        if not 0.0 <= self.test_split < 1.0:
            raise GNNTrainingConfigError(
                f"test_split must be in [0.0, 1.0), got {self.test_split}"
            )
        if self.validation_split + self.test_split >= 1.0:
            raise GNNTrainingConfigError(
                f"validation_split ({self.validation_split}) + test_split "
                f"({self.test_split}) must be < 1.0 so at least one node "
                "remains for training"
            )

        if self.patience is not None:
            _validate_int("patience", self.patience, minimum=0)
        _validate_int("eval_interval", self.eval_interval, minimum=1)

        if not isinstance(self.restore_best, bool):
            raise GNNTrainingConfigError(
                f"restore_best must be a bool, got "
                f"{type(self.restore_best).__name__}"
            )

        if not isinstance(self.loss, str):
            raise GNNTrainingConfigError(
                f"loss must be a string, got {type(self.loss).__name__}"
            )
        normalized_loss = self.loss.strip().lower()
        if normalized_loss not in SUPPORTED_LOSSES:
            raise GNNTrainingConfigError(
                f"unknown loss '{self.loss}'; expected one of "
                f"{sorted(SUPPORTED_LOSSES)}"
            )
        object.__setattr__(self, "loss", normalized_loss)

        if self.checkpoint_path is not None and not isinstance(
            self.checkpoint_path, (str, Path)
        ):
            raise GNNTrainingConfigError(
                "checkpoint_path must be a str or Path, got "
                f"{type(self.checkpoint_path).__name__}"
            )
        if isinstance(self.checkpoint_path, str):
            object.__setattr__(
                self, "checkpoint_path", Path(self.checkpoint_path)
            )

    def to_dict(self) -> dict[str, Any]:
        """Plain primitive dict (safe for checkpoints and logging)."""
        data = asdict(self)
        if data.get("checkpoint_path") is not None:
            data["checkpoint_path"] = str(data["checkpoint_path"])
        return data


def _validate_int(name: str, value: Any, *, minimum: int) -> int:
    """Reject non-integers (including ``bool``) and out-of-domain values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise GNNTrainingConfigError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    if value < minimum:
        raise GNNTrainingConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _validate_finite_float(name: str, value: Any) -> float:
    """Reject booleans, non-numbers and NaN/infinite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GNNTrainingConfigError(
            f"{name} must be a number, got {type(value).__name__}"
        )
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise GNNTrainingConfigError(
            f"{name} must be finite, got {value}"
        )
    return number


@dataclass
class TrainingHistory:
    """Structured per-epoch record of a training run (JSON-safe via to_dict).

    ``train_loss[i]`` / ``train_metrics[i]`` refer to epoch ``i + 1`` (the
    in-loop, train-mode values used for the optimizer step). ``val_loss[i]``
    / ``val_metrics[i]`` hold the post-epoch validation evaluation, or
    ``None`` for epochs where validation was skipped (``eval_interval``).
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[Optional[float]] = field(default_factory=list)
    train_metrics: list[dict[str, Optional[float]]] = field(default_factory=list)
    val_metrics: list[Optional[dict[str, Optional[float]]]] = field(
        default_factory=list
    )
    stopped_early: bool = False
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None

    @property
    def epochs_completed(self) -> int:
        """Number of completed epochs (equals ``len(train_loss)``)."""
        return len(self.train_loss)

    def to_dict(self) -> dict[str, Any]:
        """Primitive dict matching the documented history schema."""
        return {
            "train_loss": list(self.train_loss),
            "val_loss": list(self.val_loss),
            "train_metrics": [dict(m) for m in self.train_metrics],
            "val_metrics": [
                dict(m) if m is not None else None for m in self.val_metrics
            ],
            "stopped_early": self.stopped_early,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }


@dataclass(frozen=True)
class GNNCheckpoint:
    """Contents of a loaded checkpoint (see ``GNNTrainer.load_checkpoint``).

    ``model`` is a fully reconstructed, weight-populated
    :class:`~app.ml.model.GNNModel` (built from the stored ``model_config``
    unless the caller supplies an existing model to restore into).
    """

    model: GNNModel
    epoch: int
    best_val_loss: Optional[float]
    training_config: GNNTrainingConfig
    model_config: GNNConfig
    metrics: Optional[dict[str, Any]]
    history: Optional[dict[str, Any]]


class GNNTrainer:
    """Reusable training/evaluation driver around the Module 12 GNNModel.

    Responsibilities (deliberately narrow — no architecture, no dataset code):
    * validate that a Module 11 dataset can support a training run (labeled,
      non-empty, matching the model's scalar output);
    * split node indices into train/validation/test (deterministic, seeded);
    * run the node-regression training loop (loss on train nodes only);
    * evaluate loss + metrics on any node subset (eval mode, no gradients);
    * save/load state-dict based checkpoints.

    The trainer moves the given model onto the configured device (in place)
    and owns its Adam optimizer. ``train()`` returns a
    :class:`TrainingHistory`; the trained model stays available as
    ``trainer.model`` for later (Module 14+) prediction integration.
    """

    def __init__(
        self,
        model: GNNModel,
        dataset: GraphDataset,
        config: Optional[GNNTrainingConfig] = None,
    ) -> None:
        if not isinstance(model, GNNModel):
            raise GNNTrainingError(
                "model must be a Module 12 GNNModel, got "
                f"{type(model).__name__}"
            )
        if not isinstance(dataset, GraphDataset):
            raise GNNTrainingError(
                "dataset must be a Module 11 GraphDataset, got "
                f"{type(dataset).__name__}"
            )
        if config is None:
            config = GNNTrainingConfig()
        if not isinstance(config, GNNTrainingConfig):
            raise GNNTrainingConfigError(
                f"config must be a GNNTrainingConfig, got "
                f"{type(config).__name__}"
            )
        self.config = config
        self._validate_dataset_for_training(dataset, model)

        self.device = self._resolve_device(config.device)
        self.model: GNNModel = model.to(self.device)

        pyg_data = dataset.to_pyg_data()
        self.x: Tensor = pyg_data.x.to(self.device)
        self.edge_index: Tensor = pyg_data.edge_index.to(self.device)
        target = getattr(pyg_data, "y", None)
        if target is None or not bool(torch.isfinite(target).all()):
            # Defense in depth: GraphDataset validates y too, but training
            # must refuse corrupt targets even if a dataset bypassed it.
            raise GNNTrainingDataError(
                "dataset target vector is missing or contains NaN/infinite "
                "values; training requires valid finite labels"
            )
        self.y: Tensor = target.to(self.device).float()

        self.split: NodeSplit = split_nodes(
            dataset.num_nodes,
            config.validation_split,
            config.test_split,
            seed=config.seed,
        )
        self._train_mask = self._indices_to_mask(self.split.train_indices)
        self._val_mask = self._indices_to_mask(self.split.val_indices)
        self._test_mask = self._indices_to_mask(self.split.test_indices)

        self.criterion: nn.Module = SUPPORTED_LOSSES[config.loss]()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.history: Optional[TrainingHistory] = None

        logger.info(
            "GNN trainer initialized",
            extra={
                **config.to_dict(),
                "num_nodes": dataset.num_nodes,
                "num_edges": dataset.num_edges,
                "num_features": dataset.num_features,
                "split_sizes": self.split.sizes,
            },
        )

    # ------------------------------------------------------------------ #
    # Construction-time validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_dataset_for_training(
        dataset: GraphDataset, model: GNNModel
    ) -> None:
        """Reject datasets that cannot support a training run (loudly)."""
        if dataset.num_nodes < 1:
            raise GNNTrainingDataError(
                "refusing to train on an empty graph (0 nodes)"
            )
        if not dataset.has_targets or dataset.y is None:
            raise GNNTrainingDataError(
                "dataset has no targets (y is None): training requires "
                "ground-truth labels for every node. Build a labeled dataset "
                "via GraphDatasetBuilder.build(target_property=...); the "
                "builder never fabricates labels"
            )
        if dataset.y.shape != (dataset.num_nodes,):
            raise GNNTrainingDataError(
                f"target vector must have shape [{dataset.num_nodes},], got "
                f"{dataset.y.shape}"
            )
        if not np.isfinite(dataset.y.astype(np.float64)).all():
            raise GNNTrainingDataError(
                "target vector contains NaN/infinite values"
            )
        if model.config.output_dim != 1:
            raise GNNTrainingConfigError(
                "node-level scalar regression requires a model with "
                f"output_dim == 1, got {model.config.output_dim}"
            )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Parse the configured device; CUDA must exist when requested."""
        try:
            resolved = torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise GNNTrainingConfigError(
                f"device '{device}' is not a valid torch device: {exc}"
            ) from exc
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise GNNTrainingError(
                f"device '{device}' requires CUDA, which is not available; "
                "this project is CPU-first — use device='cpu'"
            )
        return resolved

    def _indices_to_mask(self, indices: np.ndarray) -> Tensor:
        """Convert split indices into a boolean node mask on the device."""
        mask = torch.zeros(
            len(self.split.train_indices)
            + len(self.split.val_indices)
            + len(self.split.test_indices),
            dtype=torch.bool,
            device=self.device,
        )
        tensor_indices = torch.as_tensor(
            indices, dtype=torch.long, device=self.device
        )
        if tensor_indices.numel():
            mask[tensor_indices] = True
        return mask

    # ------------------------------------------------------------------ #
    # Training loop (node-level regression)
    # ------------------------------------------------------------------ #

    def train(self) -> TrainingHistory:
        """Run the full training loop; returns the structured history.

        Every epoch: train-mode forward ``model(x, edge_index)`` (targets
        are NEVER an input), regression loss over TRAIN nodes only, backward,
        Adam step, then (per ``eval_interval``) an eval-mode validation pass
        whose parameters are never updated. Optionally early-stops on the
        validation loss and restores the best weights afterwards. Calling
        ``train()`` again starts a fresh history from the current weights.
        """
        cfg = self.config
        if cfg.seed is not None:
            set_seed(cfg.seed)

        history = TrainingHistory()
        best_state: Optional[dict[str, Tensor]] = None
        rounds_without_improvement = 0

        logger.info(
            "GNN training started",
            extra={**cfg.to_dict(), "split_sizes": self.split.sizes},
        )

        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_metrics = self._train_epoch()
            history.train_loss.append(train_loss)
            history.train_metrics.append(train_metrics)

            val_loss: Optional[float] = None
            val_metrics: Optional[dict[str, Optional[float]]] = None
            if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
                val_loss, val_metrics = self._evaluate_masked(self._val_mask)
                if (
                    history.best_val_loss is None
                    or val_loss < history.best_val_loss
                ):
                    history.best_val_loss = val_loss
                    history.best_epoch = epoch
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in self.model.state_dict().items()
                    }
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    if cfg.patience is not None and (
                        rounds_without_improvement >= cfg.patience
                    ):
                        history.stopped_early = True
                        logger.info(
                            "GNN training stopped early",
                            extra={
                                "epoch": epoch,
                                "best_epoch": history.best_epoch,
                                "best_val_loss": history.best_val_loss,
                            },
                        )
            history.val_loss.append(val_loss)
            history.val_metrics.append(val_metrics)
            if history.stopped_early:
                break

        if cfg.restore_best and best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(
                "GNN best model state restored",
                extra={
                    "epoch": history.best_epoch,
                    "best_val_loss": history.best_val_loss,
                },
            )

        # Attach the history BEFORE the config-driven auto-save so the saved
        # checkpoint carries the correct provenance (epoch, best validation
        # loss, metrics) instead of the "no history yet" defaults.
        self.history = history

        if cfg.checkpoint_path is not None:
            self.save_checkpoint(cfg.checkpoint_path)

        logger.info(
            "GNN training finished",
            extra={
                "epochs_run": history.epochs_completed,
                "stopped_early": history.stopped_early,
                "best_epoch": history.best_epoch,
                "best_val_loss": history.best_val_loss,
                "final_train_loss": (
                    history.train_loss[-1] if history.train_loss else None
                ),
            },
        )
        return history

    def _train_epoch(self) -> tuple[float, dict[str, Optional[float]]]:
        """One optimizer step over the train nodes; returns (loss, metrics)."""
        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.model(self.x, self.edge_index)
        loss = self.criterion(
            predictions[self._train_mask], self.y[self._train_mask]
        )
        loss.backward()
        self.optimizer.step()

        train_loss = float(loss.detach())
        if not math.isfinite(train_loss):
            raise GNNTrainingError(
                "training diverged: the loss became non-finite "
                f"({train_loss}); reduce the learning rate or inspect the "
                "dataset"
            )
        train_metrics = regression_metrics(
            predictions.detach()[self._train_mask], self.y[self._train_mask]
        )
        return train_loss, train_metrics

    # ------------------------------------------------------------------ #
    # Evaluation (eval mode, no gradients, parameters never updated)
    # ------------------------------------------------------------------ #

    def _evaluate_masked(self, mask: Tensor) -> tuple[float, dict[str, Optional[float]]]:
        """Loss + metrics over the nodes selected by a boolean mask."""
        if not bool(mask.any()):
            raise GNNEvaluationError(
                "cannot evaluate on an empty node subset"
            )
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.x, self.edge_index)
            pred_subset = predictions[mask]
            target_subset = self.y[mask]
            loss = float(self.criterion(pred_subset, target_subset))
        metrics = regression_metrics(pred_subset, target_subset)
        return loss, metrics

    def evaluate(
        self, node_indices: Optional[Sequence[int]] = None
    ) -> tuple[float, dict[str, Optional[float]]]:
        """Eval-mode (loss, metrics) over the given node indices.

        Args:
            node_indices: Node indices to evaluate (validated bounds); the
                default ``None`` evaluates every node in the graph.
        """
        if node_indices is None:
            return self._evaluate_masked(
                torch.ones(self.y.shape[0], dtype=torch.bool, device=self.device)
            )
        indices = torch.as_tensor(list(node_indices), dtype=torch.long)
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= self.y.shape[0]
        ):
            raise GNNEvaluationError(
                f"node indices out of bounds [0, {self.y.shape[0] - 1}]: "
                f"{indices.tolist()}"
            )
        if not indices.numel():
            raise GNNEvaluationError(
                "node_indices must contain at least one node index"
            )
        mask = torch.zeros(
            self.y.shape[0], dtype=torch.bool, device=self.device
        )
        mask[indices.to(self.device)] = True
        return self._evaluate_masked(mask)

    def validate(self) -> tuple[float, dict[str, Optional[float]]]:
        """Validation loss + metrics on the configured validation nodes."""
        return self._evaluate_masked(self._val_mask)

    def evaluate_test(self) -> tuple[float, dict[str, Optional[float]]]:
        """Test loss + metrics on the held-out test nodes."""
        if self.split.test_indices.size == 0:
            raise GNNTrainingDataError(
                "no test nodes: this run was configured with test_split=0.0"
            )
        return self._evaluate_masked(self._test_mask)

    def predict(self, node_indices: Optional[Sequence[int]] = None) -> Tensor:
        """Eval-mode predictions ``[num_nodes]`` (or the requested subset).

        A convenience for tests and the later prediction module — this is
        NOT a serving API and performs no HTTP/persistence work.
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.x, self.edge_index)
        if node_indices is None:
            return predictions
        indices = torch.as_tensor(list(node_indices), dtype=torch.long)
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= self.y.shape[0]
        ):
            raise GNNEvaluationError(
                f"node indices out of bounds [0, {self.y.shape[0] - 1}]: "
                f"{indices.tolist()}"
            )
        return predictions[indices.to(self.device)]

    # ------------------------------------------------------------------ #
    # Checkpointing (state_dict based, no arbitrary object pickling)
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        """Persist the training state; returns the written path.

        The file is a plain mapping of tensors and JSON-safe primitives:
        model/optimizer ``state_dict`` entries, the completed epoch, the best
        validation loss, both configs, the last validation metrics and the
        history. Large trained binaries must never be committed to Git.
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        last_val_metrics: Optional[dict[str, Any]] = None
        if self.history is not None and self.history.val_metrics:
            candidate = self.history.val_metrics[-1]
            if candidate is not None:
                last_val_metrics = dict(candidate)
        payload: dict[str, Any] = {
            _CK_FORMAT: _CHECKPOINT_FORMAT_VERSION,
            _CK_MODEL_STATE: {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.model.state_dict().items()
            },
            _CK_OPTIMIZER_STATE: self.optimizer.state_dict(),
            _CK_EPOCH: (
                self.history.epochs_completed if self.history is not None else 0
            ),
            _CK_BEST_VAL: (
                self.history.best_val_loss if self.history is not None else None
            ),
            _CK_TRAINING_CONFIG: self.config.to_dict(),
            _CK_MODEL_CONFIG: self.model.config.to_dict(),
            _CK_METRICS: last_val_metrics,
            _CK_HISTORY: (
                self.history.to_dict() if self.history is not None else None
            ),
        }
        torch.save(payload, target_path)
        logger.info(
            "GNN training checkpoint saved",
            extra={"path": str(target_path), "epoch": payload[_CK_EPOCH]},
        )
        return target_path

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        *,
        model: Optional[GNNModel] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Union[str, torch.device] = "cpu",
    ) -> GNNCheckpoint:
        """Load a checkpoint written by :meth:`save_checkpoint`.

        Args:
            path: Checkpoint file path.
            model: Optional model to restore the weights into; when omitted a
                new :class:`GNNModel` is built from the stored model config.
            optimizer: Optional optimizer whose state should be resumed.
            map_location: Device for tensor loading (CPU default — the
                project is CPU-first and checkpoints stay portable).

        Returns:
            A :class:`GNNCheckpoint` with the restored model and metadata.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            GNNCheckpointError: If the payload is not a Module 13 checkpoint
                or its stored configuration cannot be reconstructed.
        """
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"GNN training checkpoint file not found: {source_path}"
            )
        payload = torch.load(source_path, map_location=map_location, weights_only=True)
        if not isinstance(payload, Mapping):
            raise GNNCheckpointError(
                f"unrecognized GNN checkpoint format: {source_path}"
            )
        missing = [
            key
            for key in (_CK_MODEL_STATE, _CK_TRAINING_CONFIG, _CK_MODEL_CONFIG)
            if key not in payload
        ]
        if missing:
            raise GNNCheckpointError(
                f"checkpoint {source_path} is missing required key(s): {missing}"
            )

        try:
            training_config = GNNTrainingConfig(**payload[_CK_TRAINING_CONFIG])
        except (TypeError, GNNTrainingConfigError) as exc:
            raise GNNCheckpointError(
                f"checkpoint training config could not be reconstructed: {exc}"
            ) from exc
        try:
            model_config = GNNConfig(**payload[_CK_MODEL_CONFIG])
        except (TypeError, GNNModelConfigError) as exc:
            raise GNNCheckpointError(
                f"checkpoint model config could not be reconstructed: {exc}"
            ) from exc

        restored_model = model if model is not None else GNNModel.from_config(model_config)
        restored_model.load_state_dict(payload[_CK_MODEL_STATE])
        if optimizer is not None and payload.get(_CK_OPTIMIZER_STATE) is not None:
            optimizer.load_state_dict(payload[_CK_OPTIMIZER_STATE])

        return GNNCheckpoint(
            model=restored_model,
            epoch=int(payload.get(_CK_EPOCH, 0)),
            best_val_loss=payload.get(_CK_BEST_VAL),
            training_config=training_config,
            model_config=model_config,
            metrics=payload.get(_CK_METRICS),
            history=payload.get(_CK_HISTORY),
        )






