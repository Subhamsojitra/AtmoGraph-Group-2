"""Unit tests for GNN graph-data preparation (Module 11).

These tests mock the :class:`GraphRepository`, so they do NOT require a
running Neo4j server. They cover the complete Module 11 contract: extraction,
node-id -> index mapping, deterministic feature encoding, edge-index
construction/direction/filtering, target handling (no fabricated labels),
structural validation, empty-graph handling, serialization round trips,
reproducibility, and the guarded PyTorch Geometric export.

All node/target fixtures below are explicitly TEST/SYNTHETIC values.
"""

from __future__ import annotations

import math
from unittest.mock import Mock, call

import numpy as np
import pytest

from app.ml.dataset import GraphDataset, GraphDatasetBuilder
from app.ml.exceptions import (
    EmptyGraphError,
    GraphDatasetError,
    GraphDatasetValidationError,
)
from app.ml.extraction import RawEdge, RawNode, RawGraph, extract_graph
from app.ml.features import (
    RISK_LEVEL_CODES,
    UNKNOWN_LABEL,
    FeatureMetadata,
    NodeFeatureEncoder,
)
from app.repositories.graph_repository import GraphRepository
from app.services.risk.risk_service import RiskService


# --------------------------------------------------------------------------- #
# [TEST/SYNTHETIC] Fixture builders - every value here is invented test data.
# --------------------------------------------------------------------------- #


def _n(
    node_id: str,
    risk_score: float | None = None,
    risk_level: str | None = None,
    name: str = "Test",
    labels: tuple[str, ...] | None = None,
    **extra: object,
) -> dict[str, object]:
    """A repository row shaped ``{"n": {...}}`` with SYNTHETIC properties."""
    inner: dict[str, object] = {"id": node_id, "name": name}
    if risk_score is not None:
        inner["risk_score"] = risk_score
    if risk_level is not None:
        inner["risk_level"] = risk_level
    if labels is not None:
        inner["labels"] = list(labels)
    inner.update(extra)
    return {"n": inner}


def _e(source: str, target: str, rel_type: str = "SUPPLIES") -> dict[str, str]:
    """A relationship row ``(source)-[rel_type]->(target)``."""
    return {
        "source_id": source,
        "target_id": target,
        "rel_type": rel_type,
    }


def make_repo(
    nodes: list[dict[str, object]] | None = None,
    edges: list[dict[str, str]] | None = None,
) -> Mock:
    """Repository mock honoring the relationship-type whitelist like Cypher."""
    repo = Mock(spec=GraphRepository)
    repo.get_nodes.return_value = nodes or []

    def _rels(relationship_types=None, limit=50000):
        wanted = None
        if relationship_types:
            wanted = {
                t.strip() for t in relationship_types
                if isinstance(t, str) and t.strip()
            }
        rows = [
            e for e in (edges or [])
            if wanted is None or e["rel_type"] in wanted
        ]
        return rows[:limit]

    repo.find_all_relationships.side_effect = _rels
    return repo


DEFAULT_NODES = [
    _n("entity_A", risk_score=20.0, risk_level="LOW", labels=["Port"]),
    _n("entity_B", risk_score=65.0, risk_level="HIGH", labels=["Factory"]),
    _n("entity_C", risk_score=80.0, labels=["Hub"]),      # level derivable
]
DEFAULT_EDGES = [
    _e("entity_A", "entity_B"),
    _e("entity_B", "entity_C"),
]
# --------------------------------------------------------------------------- #
# 1. Graph extraction (repository reuse + parameterized call-through)
# --------------------------------------------------------------------------- #


def test_extract_graph_maps_sorted_ids_to_indices() -> None:
    raw = extract_graph(make_repo(DEFAULT_NODES, DEFAULT_EDGES))
    assert raw.node_id_to_index == {
        "entity_A": 0, "entity_B": 1, "entity_C": 2,
    }
    assert isinstance(raw.edges[0], RawEdge)


def test_extraction_is_reproducible_regardless_of_row_order() -> None:
    r1 = extract_graph(make_repo(DEFAULT_NODES, DEFAULT_EDGES))
    shuffled_nodes = list(reversed(DEFAULT_NODES))
    r2 = extract_graph(make_repo(shuffled_nodes, DEFAULT_EDGES))
    assert r1.node_id_to_index == r2.node_id_to_index
    assert r1.edges == r2.edges


def test_blank_node_ids_are_rejected() -> None:
    with pytest.raises(GraphDatasetValidationError, match="blank 'id'"):
        extract_graph(make_repo([_n("")], []))


def test_duplicate_node_ids_are_rejected() -> None:
    with pytest.raises(GraphDatasetValidationError, match="[Dd]uplicate"):
        extract_graph(make_repo(
            [_n("dup"), _n("dup")], [],
        ))


def test_edges_to_unknown_nodes_are_rejected_not_dropped() -> None:
    with pytest.raises(GraphDatasetValidationError, match="unknown"):
        extract_graph(make_repo(DEFAULT_NODES, [_e("entity_A", "ghost")]))


def test_relationship_filter_is_forwarded_to_repository() -> None:
    repo = make_repo(DEFAULT_NODES, DEFAULT_EDGES + [_e("a", "b", "UNRELATED")])
    extract_graph(repo, relationship_types=["SUPPLIES"])
    forwarded = repo.find_all_relationships.call_args.kwargs
    assert forwarded["relationship_types"] == ["SUPPLIES"]


# --------------------------------------------------------------------------- #
# 2. Node ID <-> integer index mapping
# --------------------------------------------------------------------------- #


@pytest.fixture()
def dataset() -> GraphDataset:
    return GraphDatasetBuilder(repository=make_repo(
        DEFAULT_NODES, DEFAULT_EDGES,
    )).build()


def test_node_id_to_index_mapping(dataset: GraphDataset) -> None:
    assert dataset.node_id_to_index == {
        "entity_A": 0, "entity_B": 1, "entity_C": 2,
    }


def test_index_to_node_id_round_trip(dataset: GraphDataset) -> None:
    forward = dataset.node_id_to_index
    backward = dataset.index_to_node_id
    assert all(backward[idx] == nid for nid, idx in forward.items())


def test_original_neo4j_ids_preserved_in_dataset(dataset: GraphDataset) -> None:
    assert set(dataset.node_ids) == {"entity_A", "entity_B", "entity_C"}
# --------------------------------------------------------------------------- #
# 3. Node features (numeric conversion, categorical encoding, normalization)
# --------------------------------------------------------------------------- #


def test_feature_matrix_shape_and_dtype(dataset: GraphDataset) -> None:
    assert dataset.x.shape == (3, len(NodeFeatureEncoder.FEATURE_NAMES))
    assert dataset.x.dtype == np.float32


def test_risk_score_normalized_to_unit_interval(dataset: GraphDataset) -> None:
    idx = dataset.node_id_to_index
    score_col = dataset.x[:, 0]
    assert math.isclose(float(score_col[idx["entity_A"]]), 20 / 100, abs_tol=1e-6)
    assert math.isclose(float(score_col[idx["entity_C"]]), 80 / 100, abs_tol=1e-6)


def test_missing_risk_score_gets_neutral_fill_and_counts() -> None:
    d = GraphDatasetBuilder(repository=make_repo(
        [_n("x_no_score", labels=["Port"])], [],
    )).build()
    assert float(d.x[0][0]) == 0.0
    assert d.metadata.missing_counts.get("risk_score_norm") == 1


def test_risk_level_ordinal_encoding_matches_threshold_order(
    dataset: GraphDataset,
) -> None:
    idx = dataset.node_id_to_index
    level_col = dataset.x[:, 1]
    low_code = float(level_col[idx["entity_A"]])
    high_code = float(level_col[idx["entity_B"]])
    span = len(RISK_LEVEL_CODES) - 1
    expected_low = RISK_LEVEL_CODES["LOW"] / span
    expected_high = RISK_LEVEL_CODES["HIGH"] / span
    assert low_code < high_code
    assert math.isclose(low_code, expected_low, abs_tol=1e-6)
    assert math.isclose(high_code, expected_high, abs_tol=1e-6)


def test_missing_risk_level_derived_from_score_via_existing_engine(
    dataset: GraphDataset,
) -> None:
    derived = RiskService().calculate_risk_level(80.0).upper()
    assert derived in RISK_LEVEL_CODES
    col_c = dataset.x[dataset.node_id_to_index["entity_C"], 1]
    expected = RISK_LEVEL_CODES[derived] / (len(RISK_LEVEL_CODES) - 1)
    assert math.isclose(float(col_c), expected, abs_tol=1e-6)


def test_unknown_risk_level_string_is_rejected() -> None:
    with pytest.raises(GraphDatasetValidationError, match="unknown risk_level"):
        GraphDatasetBuilder(repository=make_repo(
            [_n("odd", risk_level="BLURPLE")], [],
        )).build()


def test_label_encoded_with_deterministic_vocab_and_unknown_sentinel(
    dataset: GraphDataset,
) -> None:
    meta = dataset.metadata
    assert isinstance(meta, FeatureMetadata)
    assert meta.label_vocab[0] == UNKNOWN_LABEL
    vocab_part = sorted(list(meta.label_vocab)[1:])
    assert vocab_part == ["Factory", "Hub", "Port"]
    label_col = dataset.x[:, 2].astype(int)
    assert set(label_col.tolist()) <= set(range(len(meta.label_vocab)))


def test_out_of_range_risk_score_rejected_never_clamped() -> None:
    with pytest.raises(GraphDatasetValidationError, match="outside the valid"):
        GraphDatasetBuilder(repository=make_repo(
            [_n("hot", risk_score=150.0)], [],
        )).build()


def test_nan_feature_value_rejected() -> None:
    with pytest.raises(GraphDatasetValidationError):
        GraphDatasetBuilder(repository=make_repo(
            [_n("nan_node", risk_score=float("nan"))], [],
        )).build()


def test_boolean_score_rejected_as_non_numeric() -> None:
    bad = _n("booly")
    bad["n"]["risk_score"] = True  # type: ignore[index]
    with pytest.raises(GraphDatasetValidationError, match="must be numeric"):
        GraphDatasetBuilder(repository=make_repo([bad], [])).build()


def test_degree_features_reflect_edge_direction(dataset: GraphDataset) -> None:
    out_col = dataset.x[:, 3]
    in_col = dataset.x[:, 4]
    idx = dataset.node_id_to_index
    assert out_col.max() > 0
    assert float(out_col[idx["entity_A"]]) > float(out_col[idx["entity_C"]])
    assert float(in_col[idx["entity_C"]]) > float(in_col[idx["entity_A"]])
# --------------------------------------------------------------------------- #
# 4. Edge index: direction, filtering, shape
# --------------------------------------------------------------------------- #


def test_edge_direction_preserved_exactly_as_stored(
    dataset: GraphDataset,
) -> None:
    src, dst = dataset.edge_index
    assert [int(v) for v in src] == [0, 1]
    assert [int(v) for v in dst] == [1, 2]


def test_edge_index_shape_and_dtype(dataset: GraphDataset) -> None:
    assert dataset.edge_index.dtype == np.int64
    assert dataset.edge_index.shape == (2, 2)


def test_edge_rel_types_align_with_columns(dataset: GraphDataset) -> None:
    assert list(dataset.edge_rel_types) == ["SUPPLIES", "SUPPLIES"]
    repo = make_repo(DEFAULT_NODES, DEFAULT_EDGES + [
        _e("entity_A", "entity_C", "UNRELATED"),
    ])
    mixed = GraphDatasetBuilder(repository=repo).build()
    assert len(mixed.edge_rel_types) == 3


# --------------------------------------------------------------------------- #
# 5. Targets: explicit property only - never fabricated (Step 10/20)
# --------------------------------------------------------------------------- #


def test_unlabeled_graph_yields_no_targets(dataset: GraphDataset) -> None:
    assert dataset.y is None and not dataset.has_targets
    assert dataset.target_name is None


def test_explicit_target_property_builds_node_level_vector() -> None:
    nodes = [
        _n("a", risk_score=10.0, downstream_delay_days=2),
        _n("b", risk_score=90.0, downstream_delay_days=7),
    ]
    d = GraphDatasetBuilder(repository=make_repo(nodes, [_e("a", "b")])).build(
        target_property="downstream_delay_days",
    )
    assert d.has_targets and d.target_name == "downstream_delay_days"
    assert d.y.dtype == np.float32 and d.y.shape == (2,)
    assert [float(v) for v in d.y] == [2.0, 7.0]


def test_target_must_cover_every_node_missing_values_rejected() -> None:
    nodes = [
        _n("has_it", risk_score=10.0, downstream_delay_days=3),
        _n("lacks_it", risk_score=20.0),
    ]
    with pytest.raises(GraphDatasetValidationError, match="lack property"):
        GraphDatasetBuilder(repository=make_repo(nodes, [])).build(
            target_property="downstream_delay_days",
        )


def test_negative_and_nan_targets_rejected() -> None:
    neg = _n("neg", risk_score=10.0, downstream_delay_days=-1)
    with pytest.raises(GraphDatasetValidationError):
        GraphDatasetBuilder(repository=make_repo([neg], [])).build(
            target_property="downstream_delay_days",
        )
    nan = _n("nan", risk_score=10.0, downstream_delay_days=float("nan"))
    with pytest.raises(GraphDatasetValidationError):
        GraphDatasetBuilder(repository=make_repo([nan], [])).build(
            target_property="downstream_delay_days",
        )


def test_target_never_leaks_into_features() -> None:
    nodes = [
        _n("a", risk_score=10.0, downstream_delay_days=4),
        _n("b", risk_score=90.0, downstream_delay_days=9),
    ]
    edges = [_e("a", "b")]
    labeled = GraphDatasetBuilder(repository=make_repo(nodes, edges)).build(
        target_property="downstream_delay_days",
    )
    names = [f.lower() for f in labeled.metadata.feature_names]
    assert "downstream_delay_days" not in names
    unlabeled = GraphDatasetBuilder(repository=make_repo(nodes, edges)).build()
    assert np.array_equal(labeled.x, unlabeled.x)


def test_feature_name_cannot_be_used_as_target() -> None:
    raw = RawGraph(
        nodes=(RawNode(id="a", labels=("L",), properties={}),),
        edges=(),
        node_id_to_index={"a": 0},
    )
    builder = GraphDatasetBuilder(repository=make_repo(DEFAULT_NODES, []))
    with pytest.raises(GraphDatasetValidationError, match="leak"):
        builder.build_from_raw(raw, target_property="risk_level_code")
# --------------------------------------------------------------------------- #
# 6. Consistency validation & empty graph (Steps 15-18)
# --------------------------------------------------------------------------- #


def test_empty_graph_raises_clear_error() -> None:
    with pytest.raises(EmptyGraphError):
        GraphDatasetBuilder(repository=make_repo([], [])).build()


def test_dataset_validation_rejects_edge_index_out_of_bounds() -> None:
    with pytest.raises(GraphDatasetValidationError, match="out of bounds"):
        GraphDataset(
            node_ids=("only",), x=np.ones((1, 5), dtype=np.float32),
            edge_index=np.array([[0], [3]], dtype=np.int64),
            edge_rel_types=("X",), y=None, target_name=None, metadata=None,
        )


def test_dataset_validation_rejects_nonfinite_feature_matrix() -> None:
    x = np.ones((1, 5), dtype=np.float32)
    x[0][0] = float("inf")
    with pytest.raises(GraphDatasetValidationError, match="NaN/infinite"):
        GraphDataset(node_ids=("n",), x=x,
                     edge_index=np.zeros((2, 0), dtype=np.int64),
                     edge_rel_types=(), y=None, target_name=None,
                     metadata=None)


def test_target_vector_shape_must_match_nodes() -> None:
    with pytest.raises(GraphDatasetValidationError, match="Target vector"):
        GraphDataset(node_ids=("a", "b"), x=np.ones((2, 5), dtype=np.float32),
                     edge_index=np.zeros((2, 0), dtype=np.int64),
                     edge_rel_types=(), y=np.array([1.0]),
                     target_name="t", metadata=None)


def test_duplicate_node_ids_in_dataset_container_rejected() -> None:
    with pytest.raises(GraphDatasetValidationError, match="[Dd]uplicate"):
        GraphDataset(node_ids=("a", "a"), x=np.ones((2, 5), dtype=np.float32),
                     edge_index=np.zeros((2, 0), dtype=np.int64),
                     edge_rel_types=(), y=None, target_name=None,
                     metadata=None)


# --------------------------------------------------------------------------- #
# 7. Serialization (.npz safe round trip) + reproducibility + PyG export
# --------------------------------------------------------------------------- #


def test_npz_round_trip_preserves_everything(dataset, tmp_path) -> None:
    path = dataset.to_npz(tmp_path / "m11_unit_rt.npz")
    loaded = GraphDataset.load_npz(path)
    assert loaded.node_ids == dataset.node_ids
    assert np.array_equal(loaded.x, dataset.x)
    assert np.array_equal(loaded.edge_index, dataset.edge_index)
    assert loaded.edge_rel_types == dataset.edge_rel_types
    assert loaded.num_features == dataset.num_features
    assert loaded.num_edges == dataset.num_edges


def test_npz_archive_stores_plain_arrays_only(dataset, tmp_path) -> None:
    path = dataset.to_npz(tmp_path / "plain.npz")
    with np.load(path, allow_pickle=True) as payload:  # inspection only
        keys = set(payload.files)
    assert {"x", "edge_index", "node_ids"} <= keys
    # The loader itself must open archives in safe mode (no pickle).
    GraphDataset.load_npz(path)


def test_builder_output_is_reproducible() -> None:
    forward = GraphDatasetBuilder(repository=make_repo(
        DEFAULT_NODES, DEFAULT_EDGES,
    )).build()
    shuffled = list(reversed(DEFAULT_NODES))
    reverse = GraphDatasetBuilder(repository=make_repo(
        shuffled, DEFAULT_EDGES,
    )).build()
    assert forward.node_id_to_index == reverse.node_id_to_index
    assert np.array_equal(forward.x, reverse.x)
    assert np.array_equal(forward.edge_index, reverse.edge_index)


def test_pyg_export_raises_helpful_error_or_matches_numpy(dataset) -> None:
    try:
        data = dataset.to_pyg_data()
    except ImportError as exc:
        assert "torch_geometric" in str(exc)
        return
    import numpy as np

    assert np.array_equal(data.x.numpy(), dataset.x)
    assert np.array_equal(data.edge_index.numpy(), dataset.edge_index)
    assert data.y is None
