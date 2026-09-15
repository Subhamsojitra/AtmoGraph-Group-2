"""GNN prediction / inference layer (Module 14).

Loads a trained Module 13 checkpoint and turns it into a safe, reusable,
CPU-first inference engine over the Module 11 dataset — WITHOUT duplicating
any model, dataset or training code::

    Module 13 checkpoint (state_dict, weights_only=True)
        |   GNNTrainer.load_checkpoint (existing, reused)
        v
    GNNPredictor (this module)
        |   validates checkpoint / model config / device (loudly)
        |   model.eval() + torch.no_grad()  (always; never trains)
        v
    Module 11 GraphDataset (x, edge_index, node_ids)  -- y is NEVER an input
        v
    GNNPredictionResult  (node_id -> scalar prediction, positionally mapped)

Design decisions (documented, deliberate)
-----------------------------------------
* NO retraining and NO architecture duplication: the model is rebuilt
  exclusively through ``GNNTrainer.load_checkpoint`` (Module 13), which
  reconstructs the :class:`~app.ml.model.GNNModel` from the stored config
  and loads the stored ``state_dict`` with ``torch.load(weights_only=True)``
  (no arbitrary object deserialization).
* SCALAR node regression only: the serving contract is one predicted
  downstream-delay value per node, so checkpoints whose ``output_dim != 1``
  are rejected at load time (same constraint as Module 13 training).
* NO target leakage: ``predict(dataset)`` uses only ``x`` and
  ``edge_index`` from the dataset. The regression target ``y`` is never read.
* VALIDATION IS STRICT: missing checkpoints, corrupt checkpoints, feature
  width mismatches, empty graphs, non-finite features and non-finite outputs
  all raise dedicated :mod:`app.ml.exceptions` subclasses. Nothing is
  silently repaired.
* DEVICE: CPU-first (the development environment has no CUDA). ``device`` is
  configurable; requesting CUDA on a CPU-only machine fails loudly. The
  loaded model is always left in eval mode.
* NO HTTP, NO Neo4j, NO settings access here: this module is pure ML and is
  driven explicitly (see ``app.services.prediction_service`` for the service
  layer that wires in configuration and the Module 11 builder).

STATUS — NO ACCURACY CLAIMS: predictions come from whatever checkpoint is
deployed. The project database currently contains no real downstream-delay
labels, so no real-world predictive accuracy may be claimed anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch

from app.core.logger import get_logger
from app.ml.dataset import GraphDataset
from app.ml.exceptions import (
    GNNCheckpointError,
    GNNCheckpointInvalidError,
    GNNModelIncompatibleError,
    GNNModelInputError,
    GNNModelNotAvailableError,
    GNNPredictionInputError,
    GNNPredictionRuntimeError,
)
from app.ml.model import GNNConfig, GNNModel
from app.ml.training import GNNCheckpoint, GNNTrainer

logger = get_logger(__name__)

__all__ = ["GNNPredictor", "GNNPredictionResult"]


@dataclass(frozen=True)
class GNNPredictionResult:
    """Structured outcome of one inference run.

    ``predictions[i]`` is the scalar prediction for ``node_ids[i]`` — the
    positional mapping is guaranteed by :meth:`GNNPredictor.predict` and is
    part of the Module 14 contract (the frontend ``prediction.nodeId``
    resolves to the same graph node id).
    """

    node_ids: tuple[str, ...]
    predictions: tuple[float, ...]
    checkpoint_epoch: int

    def __post_init__(self) -> None:
        if len(self.node_ids) != len(self.predictions):
            raise GNNPredictionRuntimeError(
                "prediction result corrupted: node id count "
                f"({len(self.node_ids)}) != prediction count "
                f"({len(self.predictions)})"
            )


class GNNPredictor:
    """Reusable inference engine around a trained Module 12/13 GNN.

    Responsibilities (deliberately narrow — no architecture, no dataset, no
    training, no HTTP code):
    * load a Module 13 checkpoint safely (state_dict based, validated);
    * keep the model in eval mode on the configured device;
    * validate inference inputs against the model configuration;
    * run deterministic, gradient-free forwards and map the outputs back to
      the dataset's node ids.

    One instance owns exactly one loaded model. Construction is the only
    expensive step (checkpoint load); ``predict`` is a cheap forward pass,
    so callers should create the predictor once and reuse it (the service
    layer does exactly that).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        *,
        device: str = "cpu",
    ) -> None:
        """Load a trained checkpoint and prepare the model for inference.

        Args:
            checkpoint_path: Path to a checkpoint written by
                ``GNNTrainer.save_checkpoint`` (Module 13).
            device: Torch device for inference (``"cpu"`` default; ``"cuda"``
                only honoured when a GPU exists — checked loudly).

        Raises:
            GNNModelNotAvailableError: If the checkpoint file does not exist.
            GNNCheckpointInvalidError: If the file is corrupt or not a
                Module 13 checkpoint, or its weights do not match its stored
                architecture configuration.
            GNNModelIncompatibleError: If the stored architecture cannot
                produce the scalar per-node prediction the contract requires.
            GNNPredictionInputError: If the device specifier is invalid.
        """
        self._checkpoint_path = Path(checkpoint_path)
        self.device = self._resolve_device(device)
        self._checkpoint: GNNCheckpoint = self._load_checkpoint(
            self._checkpoint_path, self.device
        )
        self.model: GNNModel = self._checkpoint.model.to(self.device)
        self.model_config: GNNConfig = self._checkpoint.model_config
        self.epoch: int = self._checkpoint.epoch

        # Defense in depth: the Module 13 loader builds the model FROM the
        # stored config, so these must always agree. If a future checkpoint
        # format ever breaks that invariant, refuse to serve rather than
        # guessing which architecture produced the weights.
        if self.model.config != self.model_config:
            raise GNNCheckpointInvalidError(
                "checkpoint architecture mismatch: stored model config does "
                "not match the reconstructed model configuration"
            )
        if self.model_config.output_dim != 1:
            raise GNNModelIncompatibleError(
                "the prediction contract is scalar per-node regression "
                f"(output_dim == 1); the checkpoint declares "
                f"output_dim={self.model_config.output_dim}"
            )

        # Inference only: eval mode is mandatory and never reverted.
        self.model.eval()

        logger.info(
            "GNN prediction model loaded",
            extra={
                "epoch": self.epoch,
                "input_dim": self.model_config.input_dim,
                "hidden_dim": self.model_config.hidden_dim,
                "num_layers": self.model_config.num_layers,
                "device": str(self.device),
            },
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(self, dataset: GraphDataset) -> GNNPredictionResult:
        """Run eval-mode inference over the whole graph dataset.

        Args:
            dataset: A Module 11 :class:`GraphDataset`. Its regression target
                ``y`` (if any) is NEVER read — only ``x`` and ``edge_index``
                feed the model.

        Returns:
            A :class:`GNNPredictionResult` mapping every node id to its
            scalar prediction (positional, same order as
            ``dataset.node_ids``).

        Raises:
            GNNPredictionInputError: On an invalid input object, an empty
                graph, or non-finite features.
            GNNModelIncompatibleError: On a feature-width mismatch with the
                loaded model.
            GNNPredictionRuntimeError: On a forward-pass failure or a
                non-finite/mis-shaped output.
        """
        self._validate_dataset(dataset)

        pyg_data = dataset.to_pyg_data()
        x = pyg_data.x.to(self.device)
        edge_index = pyg_data.edge_index.to(self.device)

        try:
            # Eval mode + no gradients: inference never updates weights and
            # never builds autograd graphs (Module 12 dropout is inactive).
            with torch.no_grad():
                output = self.model(x, edge_index)
        except GNNModelInputError as exc:
            raise GNNPredictionInputError(
                f"model rejected the inference input: {exc}"
            ) from exc
        except Exception as exc:  # translated safely, never swallowed
            raise GNNPredictionRuntimeError(
                "GNN forward pass failed during inference"
            ) from exc

        predictions = output.detach().cpu().float()
        if predictions.ndim != 1 or predictions.shape[0] != dataset.num_nodes:
            raise GNNPredictionRuntimeError(
                "model output violates the per-node contract: expected "
                f"shape [{dataset.num_nodes}], got "
                f"{tuple(predictions.shape)}"
            )
        if not bool(torch.isfinite(predictions).all()):
            raise GNNPredictionRuntimeError(
                "model output contains NaN/infinite values; refusing to "
                "serve unusable predictions"
            )

        result = GNNPredictionResult(
            node_ids=tuple(dataset.node_ids),
            predictions=tuple(float(value) for value in predictions.tolist()),
            checkpoint_epoch=self.epoch,
        )
        logger.info(
            "GNN inference completed",
            extra={
                "num_nodes": dataset.num_nodes,
                "num_edges": dataset.num_edges,
                "epoch": self.epoch,
            },
        )
        return result

    # ------------------------------------------------------------------ #
    # Validation helpers (loud, never repairing)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Parse the configured device; CUDA must exist when requested."""
        try:
            resolved = torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise GNNPredictionInputError(
                f"device '{device}' is not a valid torch device: {exc}"
            ) from exc
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise GNNPredictionInputError(
                f"device '{device}' requires CUDA, which is not available; "
                "this project is CPU-first — use device='cpu'"
            )
        return resolved

    @staticmethod
    def _load_checkpoint(path: Path, device: torch.device) -> GNNCheckpoint:
        """Load a Module 13 checkpoint, translating failures to domain errors."""
        if not path.is_file():
            raise GNNModelNotAvailableError(
                f"trained GNN checkpoint not found: {path}"
            )
        try:
            return GNNTrainer.load_checkpoint(path, map_location=device)
        except FileNotFoundError as exc:
            # Race: the file disappeared between the check and the load.
            raise GNNModelNotAvailableError(
                f"trained GNN checkpoint not found: {path}"
            ) from exc
        except GNNCheckpointError as exc:
            raise GNNCheckpointInvalidError(
                f"checkpoint is not a valid Module 13 checkpoint: {exc}"
            ) from exc
        except Exception as exc:  # translated safely, never swallowed
            # Any other load failure (corrupt bytes, truncated file, missing
            # zip entries, weight/config mismatch) is by definition an
            # invalid checkpoint; the original error stays attached as the
            # __cause__ for log-level debugging.
            raise GNNCheckpointInvalidError(
                f"checkpoint could not be loaded safely: {exc}"
            ) from exc

    def _validate_dataset(self, dataset: GraphDataset) -> None:
        """Validate the inference input against the loaded model (loudly)."""
        if not isinstance(dataset, GraphDataset):
            raise GNNPredictionInputError(
                "dataset must be a Module 11 GraphDataset, got "
                f"{type(dataset).__name__}"
            )
        if dataset.num_nodes < 1:
            raise GNNPredictionInputError(
                "refusing to run inference on an empty graph (0 nodes)"
            )
        if dataset.num_features != self.model_config.input_dim:
            raise GNNModelIncompatibleError(
                "model was trained with input_dim="
                f"{self.model_config.input_dim} but the dataset provides "
                f"{dataset.num_features} features per node; rebuild the "
                "dataset (Module 11) or deploy a matching checkpoint"
            )
        features = dataset.x
        if features is None:
            raise GNNPredictionInputError("dataset feature matrix is missing")
        if not np.isfinite(features.astype(np.float64)).all():
            raise GNNPredictionInputError(
                "dataset features contain NaN/infinite values; corrupt "
                "inputs must fail loudly, never flow through the network"
            )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def checkpoint_path(self) -> Path:
        """Path of the checkpoint this predictor was loaded from."""
        return self._checkpoint_path