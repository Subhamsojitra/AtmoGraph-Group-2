"""GNN-ready graph dataset abstraction for AtmoGraph (Module 11).

Builds the numerical dataset consumed by the future GNN modules:

    GraphRepository (existing)
        v
    extract_graph() -> RawGraph            (app.ml.extraction)
        v
    NodeFeatureEncoder                     (app.ml.features)
        v
    GraphDatasetBuilder.build(...)         (this module)
        v
    GraphDataset(x, edge_index, y, mappings) --> Module 12 (GNN model)

Dataset layout
--------------
* ``x``          float32 ``[num_nodes, num_features]``
* ``edge_index`` int64   ``[2, num_edges]`` — row 0 = source index,
                 row 1 = target index. Direction is preserved exactly as
                 stored in Neo4j ((source)-[rel]->(target)); nothing reversed.
* ``y``          optional float32 ``[num_nodes]`` regression target.

TARGET VALUES — no fabrication policy
--------------------------------------
The AtmoGraph database currently contains NO downstream-delay/lead-time data.
Therefore:

1. By default :meth:`GraphDatasetBuilder.build` returns an **unlabeled**
   dataset (``y is None``).
2. When a numeric target property exists later (e.g. ``downstream_delay_days``
   set by an upstream ingestion process), pass its name via
   ``build(target_property=...)``. EVERY node must then carry a finite,
   non-negative number; otherwise building fails with a validation error that
   states exactly what data is missing.
3. Targets are stored ONLY in ``y``; they can never leak into ``x`` because
   the feature list of :class:`NodeFeatureEncoder` does not include them and
   the builder re-checks this at build time.
4. Synthetic target values appear ONLY inside clearly-marked TEST fixtures,
   never in production code paths.

Serialization uses ``.npz`` with ``allow_pickle=False`` in both directions —
a safe, deterministic format without arbitrary object deserialization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.core.logger import get_logger
from app.ml.exceptions import (
    EmptyGraphError,
    GraphDatasetError,
    GraphDatasetValidationError,
)
from app.ml.extraction import (
    DEFAULT_EDGE_LIMIT,
    DEFAULT_NODE_LIMIT,
    RawEdge,
    RawGraph,
    RawNode,
    extract_graph,
)
from app.ml.features import NodeFeatureEncoder
from app.repositories.graph_repository import GraphRepository
from app.services.risk.risk_service import RiskService

logger = get_logger(__name__)

_NPZ_KEY_X = "x"
_NPZ_KEY_EDGE_INDEX = "edge_index"
_NPZ_KEY_Y = "y"
_NPZ_KEY_HAS_TARGET = "has_target"
_NPZ_KEY_NODE_IDS = "node_ids"
_NPZ_KEY_REL_TYPES = "edge_rel_types"
_NPZ_KEY_METADATA = "metadata_json"



class GraphDataset:
    """Immutable, validated GNN-ready dataset (nodes + edges + optional targets).

    Instances are produced exclusively by :class:`GraphDatasetBuilder` or
    :meth:`load_npz`, which both enforce the consistency invariants
    (shapes/dtypes/index bounds) before construction succeeds.
    """

    def __init__(
        self,
        node_ids: tuple[str, ...],
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_rel_types: tuple[str, ...],
        y: Optional[np.ndarray],
        target_name: Optional[str],
        metadata: Any,
    ) -> None:
        self._validate(node_ids, x, edge_index, edge_rel_types, y)
        self._node_ids = node_ids
        self._x = np.ascontiguousarray(x, dtype=np.float32)
        self._edge_index = np.ascontiguousarray(edge_index, dtype=np.int64)
        self._edge_rel_types = edge_rel_types
        self._y = None if y is None else np.ascontiguousarray(y, dtype=np.float32)
        self._target_name = target_name
        self._metadata = metadata

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Original Neo4j node ids, ordered by their assigned index."""
        return self._node_ids

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def edge_index(self) -> np.ndarray:
        return self._edge_index

    @property
    def edge_rel_types(self) -> tuple[str, ...]:
        return self._edge_rel_types

    @property
    def y(self) -> Optional[np.ndarray]:
        """Regression target per node, ``None`` when the graph is unlabeled."""
        return self._y

    @property
    def target_name(self) -> Optional[str]:
        return self._target_name

    @property
    def metadata(self) -> Any:
        """Fitted :class:`FeatureMetadata` (or compatible mapping on load)."""
        return self._metadata

    @property
    def num_nodes(self) -> int:
        return len(self._node_ids)

    @property
    def num_edges(self) -> int:
        return int(self._edge_index.shape[1])

    @property
    def num_features(self) -> int:
        return int(self._x.shape[1])

    @property
    def has_targets(self) -> bool:
        return self._y is not None

    @property
    def node_id_to_index(self) -> dict[str, int]:
        """Mapping from the original Neo4j node id to its integer index."""
        return {node_id: idx for idx, node_id in enumerate(self._node_ids)}

    @property
    def index_to_node_id(self) -> tuple[str, ...]:
        """Inverse mapping as an ordered tuple (index -> Neo4j id)."""
        return self._node_ids

    def summary(self) -> dict[str, Any]:
        """Loggable/reportable overview (never contains credentials)."""
        feature_names = (
            tuple(self._metadata.feature_names)
            if hasattr(self._metadata, "feature_names")
            else ()
        )
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_features": self.num_features,
            "feature_names": list(feature_names),
            "has_targets": self.has_targets,
            "target_name": self.target_name,
            "directed": True,
            "format": "float32_x/int64_edge_index",
        }

    # ------------------------------------------------------------------ #
    # Serialization (.npz, allow_pickle=False both ways)
    # ------------------------------------------------------------------ #

    def to_npz(self, path: str | Path) -> Path:
        """Save deterministically to a ``.npz`` archive; returns final path."""
        target_path = Path(path)
        base_metadata: dict[str, Any] = (
            dict(self._metadata.to_dict())
            if hasattr(self._metadata, "to_dict")
            else {}
        )
        # Preserve the target name inside the metadata JSON so a round trip
        # loses no information.
        base_metadata["target_name"] = self._target_name
        np.savez(
            target_path,
            **{
                _NPZ_KEY_X: self._x,
                _NPZ_KEY_EDGE_INDEX: self._edge_index,
                _NPZ_KEY_Y: (
                    self._y if self._y is not None
                    else np.zeros(0, dtype=np.float32)
                ),
                _NPZ_KEY_HAS_TARGET: np.asarray(
                    1 if self.has_targets else 0, dtype=np.int8
                ),
                _NPZ_KEY_NODE_IDS: np.asarray(self._node_ids, dtype=np.str_),
                _NPZ_KEY_REL_TYPES: np.asarray(
                    self._edge_rel_types, dtype=np.str_
                ),
                _NPZ_KEY_METADATA: np.asarray(json.dumps(base_metadata)),
            },
        )
        logger.info(
            "GraphDataset saved",
            extra={
                "path": str(target_path),
                "num_nodes": self.num_nodes,
                "num_edges": self.num_edges,
            },
        )
        return target_path

    @classmethod
    def load_npz(cls, path: str | Path) -> "GraphDataset":
        """Load a dataset previously written by :meth:`to_npz` (safe mode)."""
        with np.load(Path(path), allow_pickle=False) as payload:
            node_ids = tuple(str(v) for v in payload[_NPZ_KEY_NODE_IDS].tolist())
            rel_types = tuple(
                str(v) for v in payload[_NPZ_KEY_REL_TYPES].tolist()
            )
            has_target = int(payload[_NPZ_KEY_HAS_TARGET]) == 1
            y = (
                np.asarray(payload[_NPZ_KEY_Y], dtype=np.float32)
                if has_target
                else None
            )
            raw_metadata: dict[str, Any] = {}
            metadata_json = str(payload[_NPZ_KEY_METADATA])
            if metadata_json.strip():
                raw_metadata = dict(json.loads(metadata_json))
            target_name: Optional[str] = raw_metadata.get("target_name")
            metadata: Any = None
            if raw_metadata:
                from app.ml.features import FeatureMetadata

                metadata = FeatureMetadata.from_dict(raw_metadata)
            loaded_y: Optional[np.ndarray] = None
            if y is not None and y.size:
                loaded_y = y
            if loaded_y is not None and not target_name:
                target_name = "target"
            return cls(
                node_ids=node_ids,
                x=np.asarray(payload[_NPZ_KEY_X], dtype=np.float32),
                edge_index=np.asarray(payload[_NPZ_KEY_EDGE_INDEX]),
                edge_rel_types=rel_types,
                y=loaded_y,
                target_name=target_name if loaded_y is not None else None,
                metadata=metadata,
            )

    # ------------------------------------------------------------------ #
    # Optional PyTorch Geometric export
    # ------------------------------------------------------------------ #

    def to_pyg_data(self) -> Any:
        """Export as ``torch_geometric.data.Data(x, edge_index, y)``.

        PyTorch Geometric is deliberately NOT a hard dependency of this
        project (see requirements.txt); when it is missing this raises an
        ``ImportError`` with installation guidance instead of failing obscurely.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is not installed; Module 11 keeps it optional. "
                "Install PyTorch Geometric matching your torch build before "
                "exporting GNN-ready Data objects (Module 12 concern)."
            ) from exc

        data_kwargs: dict[str, Any] = {
            "x": torch.from_numpy(self._x).float(),
            "edge_index": torch.from_numpy(self._edge_index).long(),
        }
        if self._y is not None:
            data_kwargs["y"] = torch.from_numpy(self._y).float()
        return Data(**data_kwargs)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate(
        node_ids: tuple[str, ...],
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_rel_types: tuple[str, ...],
        y: Optional[np.ndarray],
    ) -> None:
        """Enforce every structural invariant; raise on any violation."""
        num_nodes = len(node_ids)

        if num_nodes == 0:
            raise EmptyGraphError(
                "Refusing to build a GNN dataset with 0 nodes; produce a "
                "clear error rather than an invalid empty tensor"
            )
        if len(set(node_ids)) != num_nodes:
            raise GraphDatasetValidationError("Duplicate node ids in dataset")

        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] != num_nodes:
            raise GraphDatasetValidationError(
                f"x must have shape [num_nodes={num_nodes}, num_features], "
                f"got {x.shape}"
            )
        if x.shape[1] == 0:
            raise GraphDatasetValidationError("Feature matrix has 0 columns")
        if not np.isfinite(x.astype(np.float64)).all():
            raise GraphDatasetValidationError(
                "Feature matrix contains NaN/infinite values"
            )

        edge_index = np.asarray(edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise GraphDatasetValidationError(
                f"edge_index must have shape [2, num_edges], got "
                f"{edge_index.shape}"
            )
        if edge_index.shape[1] != len(edge_rel_types):
            raise GraphDatasetValidationError(
                "Every edge requires one relationship type "
                f"(edges={edge_index.shape[1]}, types={len(edge_rel_types)})"
            )
        if edge_index.size:
            if not np.isfinite(edge_index.astype(np.float64)).all():
                raise GraphDatasetValidationError(
                    "edge_index contains NaN/infinite values"
                )
            lo = int(edge_index.min())
            hi = int(edge_index.max())
            if lo < 0 or hi >= num_nodes:
                raise GraphDatasetValidationError(
                    f"edge_index out of bounds: valid indices are "
                    f"[0, {num_nodes - 1}], found [{lo}, {hi}]"
                )

        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.shape != (num_nodes,):
                raise GraphDatasetValidationError(
                    f"Target vector must have shape [{num_nodes},], "
                    f"got {y_arr.shape}"
                )
            if not np.isfinite(y_arr.astype(np.float64)).all():
                raise GraphDatasetValidationError(
                    "Target vector contains NaN/infinite values"
                )


class GraphDatasetBuilder:
    """Orchestrates extraction -> features -> targets -> :class:`GraphDataset`.

    Reuses the existing ``GraphRepository``/Neo4j connection end-to-end and
    performs strict validation so an invalid or inconsistent graph fails
    loudly rather than producing corrupted tensors.
    """

    def __init__(
        self,
        repository: Optional[GraphRepository] = None,
        risk_service: Optional[RiskService] = None,
        node_limit: int = DEFAULT_NODE_LIMIT,
        edge_limit: int = DEFAULT_EDGE_LIMIT,
        risk_score_min: Optional[float] = None,
        risk_score_max: Optional[float] = None,
    ) -> None:
        self._repository = repository or GraphRepository()
        self._risk_service = risk_service or RiskService()
        self._node_limit = node_limit
        self._edge_limit = edge_limit
        self._risk_score_min = risk_score_min
        self._risk_score_max = risk_score_max

    def build(
        self,
        target_property: Optional[str] = None,
        relationship_types: Optional[list[str]] = None,
    ) -> GraphDataset:
        """Build a validated, GNN-ready dataset from the live graph.

        Args:
            target_property: OPTIONAL name of the node property holding the
                regression target (e.g. future ``downstream_delay_days``).
                The project currently has no such property in production data,
                so the default (``None``) yields an unlabeled dataset with
                ``y is None``. When provided, EVERY extracted node must carry
                a finite, non-negative numeric value.
            relationship_types: Optional whitelist of relationship types that
                represent actual supply-chain propagation (Module 10's
                canonical example: ``SUPPLIES``). Omitted = all outgoing
                relationships.

        Returns:
            A fully validated :class:`GraphDataset`.

        Raises:
            EmptyGraphError: If the graph contains zero nodes.
            GraphDatasetValidationError: On any integrity violation.
        """
        try:
            raw_graph = extract_graph(
                self._repository,
                node_limit=self._node_limit,
                edge_limit=self._edge_limit,
                relationship_types=relationship_types,
            )
            if not raw_graph.nodes:
                raise EmptyGraphError(
                    "Neo4j graph contains 0 nodes; cannot prepare a GNN "
                    "dataset from an empty graph"
                )
            return self.build_from_raw(raw_graph, target_property=target_property)
        except GraphDatasetError as exc:
            logger.error(
                "GNN dataset creation failed",
                extra={"error": str(exc)},
            )
            raise


    def build_from_raw(
        self,
        raw_graph: RawGraph,
        target_property: Optional[str] = None,
    ) -> GraphDataset:
        """Assemble the dataset from an already-extracted :class:`RawGraph`.

        Split from :meth:`build` so tests (and future batch pipelines) can
        feed deterministic in-memory snapshots without touching Neo4j.
        """
        self._assert_no_target_leakage(target_property)

        encoder = NodeFeatureEncoder(
            risk_score_min=self._risk_score_min,
            risk_score_max=self._risk_score_max,
            risk_service=self._risk_service,
        )
        x = encoder.fit(raw_graph.nodes, raw_graph.edges, raw_graph.node_id_to_index)

        edge_index = np.zeros((2, 0), dtype=np.int64)
        if raw_graph.edges:
            sources = [raw_graph.node_id_to_index[e.source_id] for e in raw_graph.edges]
            targets = [raw_graph.node_id_to_index[e.target_id] for e in raw_graph.edges]
            edge_index = np.asarray([sources, targets], dtype=np.int64)

        y, target_name = self._resolve_targets(raw_graph.nodes, target_property)

        dataset = GraphDataset(
            node_ids=tuple(raw_graph.node_id_to_index.keys()),
            x=x,
            edge_index=edge_index,
            edge_rel_types=tuple(e.rel_type for e in raw_graph.edges),
            y=y,
            target_name=target_name,
            metadata=encoder.metadata,
        )

        logger.info(
            "GNN dataset created",
            extra={
                **dataset.summary(),
                "target_available": dataset.has_targets,
                "relationship_filter": None,
            },
        )
        return dataset

    # ------------------------------------------------------------------ #
    # Targets
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assert_no_target_leakage(target_property: Optional[str]) -> None:
        """Hard guarantee that a target name never appears in feature names.

        The regression target must stay separate from the input features
        (mandatory anti-leakage rule). Feature selection lives in
        ``NodeFeatureEncoder.FEATURE_NAMES``; this runtime check protects
        against future accidental additions.
        """
        if not target_property:
            return
        normalized = target_property.strip().lower()
        for feature in NodeFeatureEncoder.FEATURE_NAMES:
            if normalized == feature.lower():
                raise GraphDatasetValidationError(
                    f"Target property '{target_property}' collides with "
                    f"feature '{feature}': target values may never be used "
                    "as input features (data-leakage prevention)"
                )

    def _resolve_targets(
        self,
        nodes: tuple[RawNode, ...],
        target_property: Optional[str],
    ) -> tuple[Optional[np.ndarray], Optional[str]]:
        """Extract and validate per-node targets when a property is requested.

        Returns ``(None, None)`` for unlabeled builds. Any violation —
        missing property, boolean/non-numeric value, NaN/infinite, or a
        negative delay — aborts the build with a validation error that states
        exactly what data is required. The builder NEVER invents labels.
        """
        if not target_property:
            return None, None

        cleaned = target_property.strip()
        values = np.zeros(len(nodes), dtype=np.float32)
        missing: list[str] = []
        invalid: list[str] = []

        for idx, node in enumerate(nodes):
            raw = node.properties.get(cleaned)
            if raw is None:
                missing.append(node.id)
                continue
            try:
                number = NodeFeatureEncoder._finite_float(  # type: ignore[arg-type]
                    raw, cleaned, node.id
                )
            except GraphDatasetValidationError:
                invalid.append(node.id)
                continue
            if number < 0:
                invalid.append(node.id)
                continue
            values[idx] = number

        if missing or invalid:
            details = []
            if missing:
                sample = ", ".join(repr(m) for m in sorted(missing)[:5])
                details.append(
                    f"{len(missing)} node(s) lack property '{cleaned}' "
                    f"(sample: {sample})"
                )
            if invalid:
                sample = ", ".join(repr(i) for i in sorted(invalid)[:5])
                details.append(
                    f"{len(invalid)} node(s) have non-numeric/NaN/negative "
                    f"'{cleaned}' values (sample: {sample})"
                )
            raise GraphDatasetValidationError(
                "Target extraction failed: a labeled GNN dataset requires "
                f"an explicit finite non-negative numeric '{cleaned}' on "
                "EVERY node. Required data: " + "; ".join(details) + ". "
                "The builder does not fabricate synthetic production labels."
            )

        return values, cleaned

