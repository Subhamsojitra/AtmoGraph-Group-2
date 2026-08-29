"""Domain exceptions for the ML package (Modules 11 & 12).

Unlike the HTTP-facing modules, these exceptions have no automatic API
mapping: Modules 11/12 produce offline artifacts (datasets, model
architectures) consumed by later GNN modules, so errors must be loud and
descriptive instead of hidden behind HTTP status codes.
"""

from __future__ import annotations


class GraphDatasetError(Exception):
    """Base class for all graph-dataset preparation errors."""


class GraphDatasetValidationError(GraphDatasetError):
    """Raised when the extracted graph cannot be turned into a valid dataset.

    Examples: duplicate or missing node ids, edges pointing to unknown
    nodes, non-finite (NaN/infinite) feature or target values, inconsistent
    tensor dimensions, or out-of-domain risk scores. Failing loudly prevents
    silently corrupted training data downstream.
    """


class EmptyGraphError(GraphDatasetError):
    """Raised when the graph contains no usable nodes.

    A zero-node graph cannot produce a valid feature tensor, so callers get
    an explicit error instead of an unusable empty dataset.
    """


# --------------------------------------------------------------------------- #
# GNN model architecture (Module 12)
# --------------------------------------------------------------------------- #


class GNNModelError(Exception):
    """Base class for all GNN model-architecture errors (Module 12)."""


class GNNModelConfigError(GNNModelError):
    """Raised when the GNN architecture configuration is invalid.

    Examples: non-positive ``input_dim``/``hidden_dim``, ``num_layers < 1``,
    ``output_dim < 1``, or a dropout probability outside ``[0.0, 1.0)``.
    Failing loudly prevents silently mis-shaped models that would only
    surface much later during training.
    """


class GNNModelInputError(GNNModelError):
    """Raised when a forward-pass input violates the model contract.

    Examples: feature tensor whose width differs from ``input_dim``,
    non-float feature dtype, NaN/infinite features, an ``edge_index`` that is
    not an int64 ``[2, num_edges]`` tensor, or edge indices outside
    ``[0, num_nodes)``. The model never guesses; invalid inputs abort.
    """

# --------------------------------------------------------------------------- #
# GNN training & evaluation (Module 13)
# --------------------------------------------------------------------------- #


class GNNTrainingError(Exception):
    """Base class for all GNN training/evaluation errors (Module 13)."""


class GNNTrainingConfigError(GNNTrainingError):
    """Raised when a training configuration is invalid.

    Examples: non-positive epochs/learning rate, negative weight decay,
    split ratios outside ``[0, 1)`` or summing to ``>= 1``, a negative
    patience, an unknown loss name, or an unparseable device string.
    Failing loudly prevents silently mis-configured training runs.
    """


class GNNTrainingDataError(GNNTrainingError):
    """Raised when the dataset cannot support a training run.

    Examples: an unlabeled dataset (``y is None``), an empty graph, a
    non-finite target vector, or a node count too small for the requested
    train/validation/test split. Training requires valid labels and never
    fabricates them (Module 11 no-fabrication policy).
    """


class GNNEvaluationError(GNNTrainingError):
    """Raised when evaluation metrics or predictions cannot be computed.

    Examples: empty prediction/target tensors, mismatched shapes, out-of-
    bounds node indices, or non-finite (NaN/infinite) values. Metrics are
    never silently wrong.
    """


class GNNCheckpointError(GNNTrainingError):
    """Raised when a training checkpoint is malformed.

    Examples: a file that is not a Module 13 checkpoint (missing keys) or a
    payload whose stored configuration can no longer be reconstructed.
    """

