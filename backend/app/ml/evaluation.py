"""Node-level regression metrics for GNN evaluation (Module 13).

Metrics are computed in ``float64`` for numerical stability and are returned
under clearly named keys::

    {"mse": ..., "rmse": ..., "mae": ..., "r2": ...}

Failure policy (matching the project's fail-loudly convention):
* empty inputs, mismatched shapes, wrong dimensionality and non-finite
  (NaN/infinite) predictions/targets raise :class:`GNNEvaluationError`
  instead of producing silently wrong numbers;
* ``r2`` is ``None`` (not NaN) when it is mathematically undefined — with a
  single sample or a zero-variance (constant) target, the total sum of
  squares is zero and the score has no meaning. Consumers must treat
  ``None`` as "undefined", never as a numeric score.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from app.ml.exceptions import GNNEvaluationError

__all__ = ["regression_metrics", "METRIC_NAMES"]

#: Keys of the mapping returned by :func:`regression_metrics` (documented
#: contract used by the trainer history and checkpoints).
METRIC_NAMES: tuple[str, ...] = ("mse", "rmse", "mae", "r2")


def _to_numpy(values: Any) -> np.ndarray:
    """Coerce torch tensors (any device) or array-likes to float64 numpy."""
    if hasattr(values, "detach"):  # torch.Tensor without importing torch
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def regression_metrics(
    predictions: Any,
    targets: Any,
) -> dict[str, Optional[float]]:
    """Compute MSE / RMSE / MAE / R² over node-level regression outputs.

    Args:
        predictions: 1-D sequence/tensor of predicted values, one per node.
        targets: 1-D sequence/tensor of ground-truth values, same length.

    Returns:
        ``{"mse": float, "rmse": float, "mae": float, "r2": float | None}``.
        ``r2`` is ``None`` when undefined (fewer than 2 samples or a
        zero-variance target); every other value is a finite float.

    Raises:
        GNNEvaluationError: On empty inputs, mismatched 1-D shapes, wrong
            dimensionality, or NaN/infinite predictions or targets.
    """
    pred = _to_numpy(predictions)
    target = _to_numpy(targets)

    if pred.ndim != 1 or target.ndim != 1:
        raise GNNEvaluationError(
            "predictions/targets must be 1-D [num_nodes], got shapes "
            f"{pred.shape} and {target.shape}"
        )
    if pred.shape != target.shape:
        raise GNNEvaluationError(
            f"predictions and targets must have the same shape, got "
            f"{pred.shape} and {target.shape}"
        )
    if pred.size == 0:
        raise GNNEvaluationError(
            "cannot compute regression metrics on 0 nodes"
        )
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise GNNEvaluationError(
            "predictions/targets contain NaN/infinite values; metrics are "
            "refused instead of being silently meaningless"
        )

    error = pred - target
    mse = float(np.mean(np.square(error)))
    mae = float(np.mean(np.abs(error)))

    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(target - target.mean())))
    r2: Optional[float] = None
    if ss_tot > 0.0:
        # Defined only for >= 2 samples with a non-constant target (both
        # cases make ss_tot == 0). None is returned instead of NaN.
        r2 = 1.0 - ss_res / ss_tot

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2": r2,
    }
