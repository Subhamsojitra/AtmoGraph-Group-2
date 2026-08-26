"""Domain exceptions for graph data preparation (Module 11).

Unlike the HTTP-facing modules, these exceptions have no automatic API
mapping: Module 11 produces offline datasets consumed by the later GNN
modules, so errors must be loud and descriptive instead of hidden behind
HTTP status codes.
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
