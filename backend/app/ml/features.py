"""Deterministic node-feature engineering for the GNN dataset (Module 11).

Turns raw node properties into a fixed-size float32 feature matrix using a
fit/transform split so that parameters learned at dataset-build time can be
stored (:class:`FeatureMetadata`) and reused unchanged at inference time
(avoids data leakage from re-fitting on inference data).

INPUT FEATURES (x) vs TARGET (y)
--------------------------------
Everything encoded here is *current-state* information that exists BEFORE any
future prediction target could occur:

===================  =======================================================
feature              meaning / encoding
===================  =======================================================
risk_score_norm      Module 9 ``risk_score`` divided by the configured
                     scale maximum (0-100 domain constant, never fitted).
risk_level_code      Ordinal integer code of ``risk_level``
                     LOW=0 < MEDIUM=1 < HIGH=2 < CRITICAL=3, matching the
                     threshold order of :class:`RiskService`. Missing level
                     is derived from ``risk_score`` via the existing risk
                     engine when possible.
label_code           Integer encoding of the primary Neo4j label over a
                     deterministic, sorted vocabulary (index 0 reserved for
                     unseen labels). One-hot was rejected: the label
                     vocabulary is open-ended and would explode/shift.
out_degree_norm      Outgoing edge count / fitted max degree.
in_degree_norm       Incoming edge count / fitted max degree.
===================  =======================================================

MISSING vs INVALID policy
-------------------------
* An **absent** property is filled with a documented neutral default and
  counted in ``FeatureMetadata.missing_counts`` (surfaced in logs).
* A **present but invalid** value (NaN/infinite number, boolean masquerading
  as a score, out-of-range score, unknown risk-level string) raises
  :class:`~app.ml.exceptions.GraphDatasetValidationError` — invalid values are
  never silently coerced into training data.

None of these features use future/downstream-delay information; the
regression target stays completely separate (see ``dataset.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from app.core.logger import get_logger
from app.ml.exceptions import GraphDatasetValidationError
from app.ml.extraction import RawEdge, RawNode
from app.services.risk.risk_service import RiskService

logger = get_logger(__name__)

#: Ordered ordinal encoding for risk levels. The ORDER mirrors the documented
#: risk thresholds (LOW < MEDIUM < HIGH < CRITICAL) from Module 9 so the
#: encoding is semantically meaningful for regression models.
RISK_LEVEL_CODES: dict[str, int] = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}

#: Sentinel vocabulary entry (code 0) for labels unseen during fit or nodes
#: without any Neo4j label.
UNKNOWN_LABEL = "<UNKNOWN>"


@dataclass(frozen=True)
class FeatureMetadata:
    """Fitted encoder parameters persisted alongside every dataset.

    Storing these values with the dataset guarantees that inference-time
    transformations apply exactly the same mappings/scaling as build time
    (Step 7 requirement: learned parameters must be reusable deterministically).

    ``missing_counts`` records how many fill-value imputations were applied per
    feature during fit/transform, providing transparency instead of silent
    corruption of the feature matrix.
    """

    feature_names: tuple[str, ...]
    risk_score_min: float
    risk_score_max: float
    label_vocab: tuple[str, ...]
    degree_max_out: float
    degree_max_in: float
    missing_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary representation (used by .npz serialization)."""
        return {
            "feature_names": list(self.feature_names),
            "risk_score_min": self.risk_score_min,
            "risk_score_max": self.risk_score_max,
            "label_vocab": list(self.label_vocab),
            "degree_max_out": self.degree_max_out,
            "degree_max_in": self.degree_max_in,
            "missing_counts": dict(self.missing_counts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureMetadata":
        """Rebuild metadata from :meth:`to_dict` output."""
        return cls(
            feature_names=tuple(payload["feature_names"]),
            risk_score_min=float(payload["risk_score_min"]),
            risk_score_max=float(payload["risk_score_max"]),
            label_vocab=tuple(payload["label_vocab"]),
            degree_max_out=float(payload["degree_max_out"]),
            degree_max_in=float(payload["degree_max_in"]),
            missing_counts={str(k): int(v) for k, v in payload[
                "missing_counts"
            ].items()},
        )


class NodeFeatureEncoder:
    """Fit/transform encoder producing the node feature matrix.

    The same instance must be used to transform inference graphs that were fit
    on the training graph; the fitted :class:`FeatureMetadata` travels with the
    serialized dataset so Module 12 can rebuild an identical encoder.
    """

    #: Fixed, documented feature order — DO NOT reorder without bumping the
    #: dataset format, as serialized matrices depend on this layout.
    FEATURE_NAMES: tuple[str, ...] = (
        "risk_score_norm",
        "risk_level_code",
        "label_code",
        "out_degree_norm",
        "in_degree_norm",
    )

    def __init__(
        self,
        risk_score_min: Optional[float] = None,
        risk_score_max: Optional[float] = None,
        risk_service: Optional[RiskService] = None,
    ) -> None:
        # Defaults come from the shared settings so the encoder always agrees
        # with Modules 9/10 about the valid score domain.
        from app.core.config import settings

        self._score_min = (
            float(settings.risk_score_min)
            if risk_score_min is None
            else float(risk_score_min)
        )
        self._score_max = (
            float(settings.risk_score_max)
            if risk_score_max is None
            else float(risk_score_max)
        )
        if not self._score_max > self._score_min:
            raise ValueError(
                "risk_score_max must be strictly greater than risk_score_min"
            )
        # Reuse the EXISTING risk engine for level derivation (no duplication).
        self._risk_service = risk_service or RiskService()

        self._metadata: Optional[FeatureMetadata] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def metadata(self) -> FeatureMetadata:
        """Fitted parameters; only available after :meth:`fit`."""
        if self._metadata is None:
            raise RuntimeError("NodeFeatureEncoder.fit() must run before use")
        return self._metadata

    def compute_degrees(
        self, edges: tuple[RawEdge, ...], num_nodes: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(out_degree, in_degree)`` arrays in index space.

        Uses the project-wide relationship direction convention: an edge
        ``(source)->(target)`` increments the source's out-degree and the
        target's in-degree.
        """
        out_deg = np.zeros(num_nodes, dtype=np.float64)
        in_deg = np.zeros(num_nodes, dtype=np.float64)
        index_map = self._index_map
        for edge in edges:
            src = index_map[edge.source_id]
            dst = index_map[edge.target_id]
            out_deg[src] += 1.0
            in_deg[dst] += 1.0
        return out_deg, in_deg

    def fit(
        self,
        nodes: tuple[RawNode, ...],
        edges: tuple[RawEdge, ...],
        index_map: Mapping[str, int],
    ) -> np.ndarray:
        """Fit encoder parameters and return the feature matrix.

        Args:
            nodes: Nodes in deterministic (sorted-id) order.
            edges: Validated edges whose endpoints are inside ``index_map``.
            index_map: ``node_id -> row index`` mapping from extraction.

        Returns:
            float32 array of shape ``[len(nodes), len(FEATURE_NAMES)]``.
        """
        self._index_map: Mapping[str, int] = dict(index_map)

        vocab = {UNKNOWN_LABEL: 0}
        next_code = 1
        for node in nodes:
            label = node.primary_label or UNKNOWN_LABEL
            if label not in vocab:
                vocab[label] = next_code
                next_code += 1

        missing_counts: dict[str, int] = {}

        x = self._encode(nodes, edges, vocab, missing_counts)

        out_deg, in_deg = self.compute_degrees(edges, len(nodes))
        deg_cap_out = float(out_deg.max()) if len(nodes) else 0.0
        deg_cap_in = float(in_deg.max()) if len(nodes) else 0.0

        self._metadata = FeatureMetadata(
            feature_names=self.FEATURE_NAMES,
            risk_score_min=self._score_min,
            risk_score_max=self._score_max,
            label_vocab=tuple(sorted(vocab)),
            degree_max_out=deg_cap_out,
            degree_max_in=deg_cap_in,
            missing_counts=missing_counts,
        )

        self.log_fit_summary(x)
        return x

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def log_fit_summary(self, x: np.ndarray) -> None:
        """Emit one structured summary line for a completed fit."""
        assert self._metadata is not None
        vocab_size = len(self._metadata.label_vocab)
        missing_total = sum(self._metadata.missing_counts.values())
        logger.info(
            "Node feature encoding complete",
            extra={
                "nodes": int(x.shape[0]),
                "features": int(x.shape[1]),
                "label_vocabulary": vocab_size,
                "missing_feature_fills": missing_total,
                "target_available": False,
            },
        )

    def _encode(
        self,
        nodes: tuple[RawNode, ...],
        edges: tuple[RawEdge, ...],
        vocab: Mapping[str, int],
        missing_counts: dict[str, int],
    ) -> np.ndarray:
        """Encode every node into one float32 row of fixed width."""
        out_deg, in_deg = self.compute_degrees(edges, len(nodes))
        deg_cap_out = self._degree_cap(edges, nodes, direction="out")
        deg_cap_in = self._degree_cap(edges, nodes, direction="in")

        rows = np.zeros((len(nodes), len(self.FEATURE_NAMES)), dtype=np.float32)

        for idx, node in enumerate(nodes):
            props = node.properties

            score = self._resolve_risk_score(props, node.id, missing_counts)
            level_code = self._resolve_level_code(props, score, node.id, missing_counts)
            label_code = vocab.get(node.primary_label or UNKNOWN_LABEL, 0)
            if node.primary_label and node.primary_label not in vocab:
                # Unseen label at transform time -> sentinel + transparency.
                self._count(missing_counts, "label_code")

            scale_span = self._score_max - self._score_min
            rows[idx, 0] = (score - self._score_min) / scale_span
            rows[idx, 1] = float(level_code) / float(len(RISK_LEVEL_CODES) - 1)
            rows[idx, 2] = float(label_code)
            rows[idx, 3] = out_deg[idx] / deg_cap_out
            rows[idx, 4] = in_deg[idx] / deg_cap_in

        return rows

    def _degree_cap(
        self,
        edges: tuple[RawEdge, ...],
        nodes: tuple[RawNode, ...],
        direction: str,
    ) -> float:
        """Scaling denominator for degree features.

        Uses the value FROZEN by fit when available so inference scaling never
        drifts; falls back to this graph's own maximum during fitting.
        """
        if self._metadata is not None:
            cap = (
                self._metadata.degree_max_out
                if direction == "out"
                else self._metadata.degree_max_in
            )
        else:
            out_deg, in_deg = self.compute_degrees(edges, len(nodes))
            degrees = out_deg if direction == "out" else in_deg
            cap = float(degrees.max()) if degrees.size else 0.0
        return cap if cap > 0 else 1.0  # all-zero degrees divide safely by 1

    def _resolve_risk_score(
        self,
        props: Mapping[str, Any],
        node_id: str,
        missing_counts: dict[str, int],
    ) -> float:
        """Extract a validated risk score on the configured domain."""
        raw = props.get("risk_score")
        if raw is None:
            self._count(missing_counts, "risk_score_norm")
            return self._score_min  # neutral fill: the lowest possible risk

        score = self._finite_float(raw, "risk_score", node_id)
        if not self._score_min <= score <= self._score_max:
            raise GraphDatasetValidationError(
                f"Node '{node_id}' has risk_score {score} outside the valid "
                f"[{self._score_min}, {self._score_max}] range; out-of-range "
                "scores are rejected (never clamped), matching Module 9 policy"
            )
        return score

    def _resolve_level_code(
        self,
        props: Mapping[str, Any],
        score: Optional[float],
        node_id: str,
        missing_counts: dict[str, int],
    ) -> int:
        """Resolve the ordinal risk-level code.

        Preference order: persisted ``risk_level`` property > derivation from
        ``risk_score`` via the existing :class:`RiskService` thresholds >
        neutral LOW fill with an imputation count.
        """
        raw_level = props.get("risk_level")
        if isinstance(raw_level, str) and raw_level.strip():
            normalized = raw_level.strip().upper()
            if normalized not in RISK_LEVEL_CODES:
                raise GraphDatasetValidationError(
                    f"Node '{node_id}' has unknown risk_level "
                    f"'{raw_level.strip()}'; expected one of "
                    f"{sorted(RISK_LEVEL_CODES)}"
                )
            return RISK_LEVEL_CODES[normalized]

        derived_raw = getattr(score, "real", None)
        if isinstance(derived_raw, float) and props.get("risk_score") is not None:
            level = self._risk_service.calculate_risk_level(float(derived_raw))
            normalized = str(level).strip().upper()
            if normalized in RISK_LEVEL_CODES:
                return RISK_LEVEL_CODES[normalized]

        self._count(missing_counts, "risk_level_code")
        return RISK_LEVEL_CODES["LOW"]

    @staticmethod
    def _finite_float(value: Any, feature: str, node_id: str) -> float:
        """Coerce ``value`` to a finite float or raise a validation error.

        Booleans are rejected explicitly (``bool`` subclasses ``int`` but is
        semantically not a measurement), as are NaN/infinite floats.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GraphDatasetValidationError(
                f"Node '{node_id}' feature '{feature}' must be numeric, "
                f"got {type(value).__name__}"
            )
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise GraphDatasetValidationError(
                f"Node '{node_id}' feature '{feature}' is non-finite "
                f"(NaN/inf): corrupt values must fail loudly, never enter tensors"
            )
        return number

    @staticmethod
    def _count(missing_counts: dict[str, int], feature: str) -> None:
        missing_counts[feature] = missing_counts.get(feature, 0) + 1

    def transform(
        self,
        nodes: tuple[RawNode, ...],
        edges: tuple[RawEdge, ...],
        index_map: Mapping[str, int],
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Encode new data using parameters frozen by a previous :meth:`fit`.

        Returns the matrix plus per-call missing-feature counts so callers can
        log imputation transparency without mutating fitted metadata.
        """
        if self._metadata is None:
            raise RuntimeError("NodeFeatureEncoder.fit() must run before transform()")
        self._index_map = dict(index_map)
        vocab = {name: code for code, name in enumerate(self._metadata.label_vocab)}
        missing_counts: dict[str, int] = {}
        x = self._encode(nodes, edges, vocab, missing_counts)
        return x, missing_counts

