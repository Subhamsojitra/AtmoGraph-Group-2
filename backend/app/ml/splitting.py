"""Deterministic node splitting for node-level GNN training (Module 13).

Module 11 produces ONE graph (a single supply-chain snapshot) and Module 12's
model consumes one ``(x, edge_index)`` pair covering the whole graph.
Node-level regression on a single graph is therefore TRANSDUCTIVE (the classic
Kipf & Welling semi-supervised setup): every node flows through the model,
while the train/validation/test split selects WHICH nodes' ground-truth
targets contribute to the loss and to the metrics.

Why this split strategy is safe (leakage analysis)
--------------------------------------------------
* Targets are used ONLY for the loss/metrics of their own split. They are
  never model inputs: ``GNNModel.forward(x, edge_index)`` has no target
  parameter (Module 12 contract) and the trainer never passes ``y`` to it.
* Module 11 guarantees the feature matrix contains no target information
  (``GraphDatasetBuilder._assert_no_target_leakage``), so messages passed
  along ``edge_index`` carry current-state features only. Validation/test
  target values therefore cannot enter training through message passing.
* An inductive subgraph split (dropping edges whose endpoint lies outside
  the training nodes) is intentionally NOT used: Module 11 defines one
  connected supply-chain graph, and removing edges would change the very
  message-passing structure the model must learn on.

The split is fully deterministic for a given ``seed``: it uses a locally
seeded ``numpy.random.Generator`` and never touches interpreter-global RNG
state (``random`` / ``numpy.random``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.ml.exceptions import GNNTrainingConfigError, GNNTrainingDataError

__all__ = ["NodeSplit", "split_nodes"]


@dataclass(frozen=True)
class NodeSplit:
    """Deterministic partition of node indices into train/validation/test.

    Each field is a 1-D int64 array of DISTINCT node indices. The three
    arrays are pairwise disjoint and their union is exactly
    ``[0, num_nodes)`` — no node is ever silently dropped. Empty arrays are
    valid (a split ratio of ``0.0`` yields an empty partition).
    """

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def sizes(self) -> tuple[int, int, int]:
        """``(num_train, num_val, num_test)`` node counts."""
        return (
            int(self.train_indices.size),
            int(self.val_indices.size),
            int(self.test_indices.size),
        )


def _validate_ratio(name: str, value: object) -> float:
    """Reject booleans, non-numbers, NaN and values outside ``[0.0, 1.0)``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GNNTrainingConfigError(
            f"{name} must be a float, got {type(value).__name__}"
        )
    ratio = float(value)
    if np.isnan(ratio) or ratio < 0.0 or ratio >= 1.0:
        raise GNNTrainingConfigError(
            f"{name} must be in [0.0, 1.0), got {value}"
        )
    return ratio


def split_nodes(
    num_nodes: int,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    seed: Optional[int] = None,
) -> NodeSplit:
    """Split ``num_nodes`` node indices into train/validation/test subsets.

    Counts are computed from the ratios with rounding, then floored at one
    node for every non-zero ratio so that a requested split is never
    silently dropped, and every non-requested node goes to training (the
    three counts always sum to ``num_nodes`` — no data is lost).

    Args:
        num_nodes: Total number of graph nodes (>= 1).
        validation_split: Fraction of nodes for validation, in ``[0, 1)``.
        test_split: Fraction of nodes for the held-out test set, in ``[0, 1)``.
        seed: Optional seed for the shuffling. The same seed always produces
            the same split for the same ``num_nodes``; ``None`` draws from
            OS entropy (non-deterministic).

    Raises:
        GNNTrainingDataError: If ``num_nodes`` is not a positive integer, or
            the dataset is too small to satisfy the requested split while
            keeping at least one training node.
        GNNTrainingConfigError: If a ratio is malformed, or
            ``validation_split + test_split >= 1``.
    """
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
        raise GNNTrainingDataError(
            f"num_nodes must be an integer, got {type(num_nodes).__name__}"
        )
    if num_nodes < 1:
        raise GNNTrainingDataError(
            f"cannot split a dataset with {num_nodes} node(s); a training "
            "run requires at least one node"
        )

    val_ratio = _validate_ratio("validation_split", validation_split)
    test_ratio = _validate_ratio("test_split", test_split)
    if val_ratio + test_ratio >= 1.0:
        raise GNNTrainingConfigError(
            f"validation_split ({validation_split}) + test_split "
            f"({test_split}) must be < 1.0 so at least one node remains "
            "for training"
        )

    n_val = max(1, int(round(num_nodes * val_ratio))) if val_ratio > 0 else 0
    n_test = max(1, int(round(num_nodes * test_ratio))) if test_ratio > 0 else 0
    n_train = num_nodes - n_val - n_test
    if n_train < 1:
        raise GNNTrainingDataError(
            f"dataset too small for the requested split: {num_nodes} node(s) "
            f"cannot provide {n_val} validation + {n_test} test node(s) and "
            "still leave at least 1 training node"
        )

    generator = np.random.default_rng(seed)
    permutation = generator.permutation(num_nodes)
    return NodeSplit(
        train_indices=permutation[:n_train].astype(np.int64),
        val_indices=permutation[n_train:n_train + n_val].astype(np.int64),
        test_indices=permutation[n_train + n_val:].astype(np.int64),
    )
