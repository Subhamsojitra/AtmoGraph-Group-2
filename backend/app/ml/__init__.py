"""Machine-learning data preparation package for AtmoGraph (Module 11).

This package turns the existing Neo4j supply-chain graph into a numerical,
GNN-ready dataset. Its scope is deliberately limited to DATA PREPARATION:

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
                                           .npz serialization, optional
                                           PyTorch Geometric export)

Feeding Module 12 (GNN model) happens LATER; nothing in this package trains,
evaluates, or serves a model.

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
synthetic targets are confined to clearly-marked test fixtures.
"""

from __future__ import annotations

from app.ml.dataset import GraphDataset, GraphDatasetBuilder
from app.ml.exceptions import (
    EmptyGraphError,
    GraphDatasetError,
    GraphDatasetValidationError,
)
from app.ml.extraction import RawEdge, RawGraph, RawNode, extract_graph
from app.ml.features import FeatureMetadata, NodeFeatureEncoder

__all__ = [
    "GraphDataset",
    "GraphDatasetBuilder",
    "EmptyGraphError",
    "GraphDatasetError",
    "GraphDatasetValidationError",
    "RawEdge",
    "RawGraph",
    "RawNode",
    "extract_graph",
    "FeatureMetadata",
    "NodeFeatureEncoder",
]
