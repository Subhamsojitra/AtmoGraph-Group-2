"""Machine-learning package for AtmoGraph (Modules 11, 12, 13 & 14).

Module 11 turns the existing Neo4j supply-chain graph into a numerical,
GNN-ready dataset. Module 12 adds the GNN model architecture that consumes
it. Module 13 adds training and evaluation around both. Module 14 adds the
prediction/inference layer that serves the trained model. Scope of this
package:

    Existing Neo4j Graph
        |   GraphRepository.get_nodes / find_all_relationships
        |   (parameterized Cypher only, no interpolated values)
        v
    Graph extraction                      (app.ml.extraction)
        v
    Node-ID -> integer-index mapping      (app.ml.extraction.RawGraph)
        v
    Node features                         (app.ml.features.NodeFeatureEncoder)
        v
    Edge index / targets / validation     (app.ml.dataset.GraphDatasetBuilder)
        v
    GNN-ready GraphDataset                (app.ml.dataset.GraphDataset,
                                           .npz serialization, PyTorch
                                           Geometric export)
        v
    GNN model architecture                (app.ml.model.GNNModel — Module 12,
                                           node-level regression)
        v
    Training & evaluation                 (app.ml.training.GNNTrainer —
                                           Module 13: seeded/deterministic
                                           node splits, Adam, regression
                                           loss on train nodes, validation,
                                           metrics, checkpoints; NO serving
                                           API — later modules integrate
                                           predictions)
        |   app.ml.splitting  (deterministic NodeSplit)
        |   app.ml.evaluation (MSE / RMSE / MAE / R2)

INPUT FEATURES vs TARGET (leakage policy)
-----------------------------------------
The current node-feature vector is::

    risk_score_norm     current Module 9 risk score rescaled to [0, 1]
    risk_level_code     ordinal encoding of LOW/MEDIUM/HIGH/CRITICAL
    label_code          integer encoding of the primary Neo4j label
    out_degree_norm     outgoing edge count scaled by the fitted maximum
    in_degree_norm      incoming edge count scaled by the fitted maximum

The prediction TARGET (node regression, e.g. future ``downstream_delay_days``)
is handled completely separately: it is passed explicitly via
``GraphDatasetBuilder.build(target_property=...)``, stored only in ``y``,
validated independently, and can never end up inside ``x``. The project
currently contains NO real target values anywhere in the database, so callers
get an unlabeled dataset (``y is None``) until such a property actually exists;
synthetic targets are confined to clearly-marked test fixtures. Neither the
Module 12 model (``forward(x, edge_index)``) nor the Module 13 trainer ever
receive ``y`` as an input — targets are used only for the loss and metrics
over their own train/validation/test node split.
"""

from __future__ import annotations

from app.ml.dataset import GraphDataset, GraphDatasetBuilder
from app.ml.evaluation import METRIC_NAMES, regression_metrics
from app.ml.exceptions import (
    EmptyGraphError,
    GNNCheckpointError,
    GNNCheckpointInvalidError,
    GNNEvaluationError,
    GNNModelConfigError,
    GNNModelError,
    GNNModelIncompatibleError,
    GNNModelInputError,
    GNNModelNotAvailableError,
    GNNPredictionError,
    GNNPredictionInputError,
    GNNPredictionRuntimeError,
    GNNTrainingConfigError,
    GNNTrainingDataError,
    GNNTrainingError,
    GraphDatasetError,
    GraphDatasetValidationError,
)
from app.ml.extraction import RawEdge, RawGraph, RawNode, extract_graph
from app.ml.features import FeatureMetadata, NodeFeatureEncoder
from app.ml.model import GNNConfig, GNNModel
from app.ml.prediction import GNNPredictionResult, GNNPredictor
from app.ml.splitting import NodeSplit, split_nodes
from app.ml.training import (
    GNNCheckpoint,
    GNNTrainingConfig,
    GNNTrainer,
    TrainingHistory,
    set_seed,
)

__all__ = [
    "GraphDataset",
    "GraphDatasetBuilder",
    "EmptyGraphError",
    "GraphDatasetError",
    "GraphDatasetValidationError",
    "GNNConfig",
    "GNNModel",
    "GNNModelError",
    "GNNModelConfigError",
    "GNNModelInputError",
    "GNNTrainingError",
    "GNNTrainingConfigError",
    "GNNTrainingDataError",
    "GNNEvaluationError",
    "GNNCheckpointError",
    "GNNCheckpoint",
    "GNNTrainingConfig",
    "GNNTrainer",
    "TrainingHistory",
    "GNNPredictionError",
    "GNNModelNotAvailableError",
    "GNNCheckpointInvalidError",
    "GNNModelIncompatibleError",
    "GNNPredictionInputError",
    "GNNPredictionRuntimeError",
    "GNNPredictor",
    "GNNPredictionResult",
    "METRIC_NAMES",
    "NodeSplit",
    "RawEdge",
    "RawGraph",
    "RawNode",
    "extract_graph",
    "FeatureMetadata",
    "NodeFeatureEncoder",
    "regression_metrics",
    "set_seed",
    "split_nodes",
]

