"""Real Neo4j integration tests for GNN graph-data preparation (Module 11).

These tests execute against a REAL Neo4j server and are intentionally skipped
when it is unreachable (same pattern as Module 10's ``test_risk_propagation_
neo4j.py``): nothing is mocked. All created nodes use a dedicated test label
and unique ids, and are deleted afterwards.

Run from the ``backend`` directory::

    python -m pytest tests/test_gnn_dataset_neo4j.py -v
"""

from __future__ import annotations

import math
import uuid

import numpy as np
import pytest

from app.database.neo4j import Neo4jDatabase
from app.ml.dataset import GraphDataset, GraphDatasetBuilder
from app.ml.exceptions import EmptyGraphError, GraphDatasetValidationError
from app.repositories.graph_repository import GraphRepository

_TEST_LABEL = "TestGNNDataset"
_TARGET_PROP = "downstream_delay_days_test_synthetic"


def _require_neo4j(db: Neo4jDatabase) -> None:
    """Skip when a real Neo4j server is not reachable."""
    if not db.verify_connectivity():
        pytest.skip(
            "Neo4j is not available - start Neo4j (neo4j://localhost:7687) "
            "before running the real integration test"
        )


def _uid(prefix: str) -> str:
    return f"test-gnn-{prefix}-{uuid.uuid4().hex[:8]}"


def _create_node(
    db: Neo4jDatabase,
    node_id: str,
    name: str,
    risk_score: float | None = None,
    risk_level: str | None = None,
    target_days: float | None = None,
) -> None:
    """Create one TEST-labeled node with optional synthetic properties."""
    clauses = ["id: $id", "name: $name"]
    params: dict[str, object] = {"id": node_id, "name": name}
    if risk_score is not None:
        clauses.append("risk_score: $risk_score")
        params["risk_score"] = risk_score
    if risk_level is not None:
        clauses.append("risk_level: $risk_level")
        params["risk_level"] = risk_level
    if target_days is not None:
        clauses.append(f"{_TARGET_PROP}: $target_days")
        params["target_days"] = target_days
    query = f"CREATE (n:`{_TEST_LABEL}` {{{', '.join(clauses)}}})"
    with db.get_session() as session:
        session.run(query, params).consume()


def _create_relationship(
    db: Neo4jDatabase, source_id: str, target_id: str, rel_type: str
) -> None:
    query = (
        f"MATCH (a:`{_TEST_LABEL}` {{id: $source}}) "
        f"MATCH (b:`{_TEST_LABEL}` {{id: $target}}) "
        f"CREATE (a)-[:`{rel_type}`]->(b)"
    )
    with db.get_session() as session:
        session.run(query, {"source": source_id, "target": target_id}).consume()


def _delete_by_ids(db: Neo4jDatabase, node_ids: list[str]) -> None:
    for node_id in node_ids:
        try:
            with db.get_session() as session:
                session.run(
                    f"MATCH (n:`{_TEST_LABEL}` {{id: $id}}) DETACH DELETE n",
                    {"id": node_id},
                ).consume()
        except Exception:
            pass

def _owned_pairs(
    dataset: GraphDataset, mapped: dict[str, int], owned: set[str]
) -> set[tuple[int, int]]:
    """Edge pairs whose BOTH endpoints belong to this test's created nodes."""
    index_to_id = {idx: nid for nid, idx in mapped.items()}
    src, dst = dataset.edge_index
    n = len(index_to_id)
    return {
        (int(s), int(t))
        for s, t in zip(src.tolist(), dst.tolist())
        if int(s) < n and int(t) < n
        and index_to_id.get(int(s)) in owned
        and index_to_id.get(int(t)) in owned
    }


# --------------------------------------------------------------------------- #
# 1. Real extraction: nodes, edges, direction, features from persisted props
# --------------------------------------------------------------------------- #


def test_real_extraction_edges_direction_and_features() -> None:
    db = Neo4jDatabase()
    ids = [_uid("a"), _uid("b"), _uid("c")]
    try:
        db.initialize()
        _require_neo4j(db)
        _create_node(db, ids[0], "Port X", risk_score=30.0, risk_level="LOW")
        _create_node(db, ids[1], "Factory Y", risk_score=70.0, risk_level="HIGH")
        _create_node(db, ids[2], "Hub Z", risk_score=95.0, risk_level="CRITICAL")
        _create_relationship(db, ids[0], ids[1], "SUPPLIES")
        _create_relationship(db, ids[1], ids[2], "SUPPLIES")

        repository = GraphRepository()
        dataset = GraphDatasetBuilder(repository=repository).build(
            relationship_types=["SUPPLIES"],
        )

        mapped = {nid: dataset.node_id_to_index[nid] for nid in ids}
        owned = _owned_pairs(dataset, mapped, set(ids))
        pairs = {(mapped[ids[0]], mapped[ids[1]]),
                 (mapped[ids[1]], mapped[ids[2]])}
        assert pairs <= owned, f"expected supply edges missing: {pairs} vs {owned}"
        assert (mapped[ids[1]], mapped[ids[0]]) not in owned

        row_a = dataset.x[mapped[ids[0]]]
        row_c = dataset.x[mapped[ids[2]]]
        scale = (dataset.metadata.risk_score_max
                 - dataset.metadata.risk_score_min)
        min_v = dataset.metadata.risk_score_min
        assert math.isclose(float(row_a[0]) * scale + min_v, 30.0, abs_tol=1e-3)
        assert math.isclose(float(row_c[0]) * scale + min_v, 95.0, abs_tol=1e-3)
        assert float(dataset.x[mapped[ids[2]], 1]) > float(row_a[0, 1])
    finally:
        _delete_by_ids(db, ids)
        db.close()


# --------------------------------------------------------------------------- #
# 2. Real relationship-type filtering excludes unrelated relationship types
# --------------------------------------------------------------------------- #


def test_real_relationship_type_filtering() -> None:
    db = Neo4jDatabase()
    ids = [_uid("f1"), _uid("f2"), _uid("f3")]
    try:
        db.initialize()
        _require_neo4j(db)
        _create_node(db, ids[0], "A", risk_score=10.0)
        _create_node(db, ids[1], "B", risk_score=20.0)
        _create_node(db, ids[2], "C", risk_score=30.0)
        _create_relationship(db, ids[0], ids[1], "SUPPLIES")
        _create_relationship(db, ids[0], ids[2], "LOCATED_IN")  # not supply-chain

        repository = GraphRepository()

        def count_between(ds: GraphDataset, s_id: str, t_id: str) -> int:
            mapping = ds.node_id_to_index
            if s_id not in mapping or t_id not in mapping:
                return 0
            s_i, t_i = mapping[s_id], mapping[t_id]
            src, dst = ds.edge_index
            return sum(
                1 for s, t in zip(src.tolist(), dst.tolist())
                if s == s_i and t == t_i
            )

        filtered = GraphDatasetBuilder(repository=repository).build(
            relationship_types=["SUPPLIES"],
        )
        unfiltered = GraphDatasetBuilder(repository=repository).build()

        assert count_between(filtered, ids[0], ids[1]) >= 1
        assert count_between(filtered, ids[0], ids[2]) == 0
        assert count_between(unfiltered, ids[0], ids[2]) >= 1
    finally:
        _delete_by_ids(db, ids)
        db.close()


# --------------------------------------------------------------------------- #
# 3. Synthetic target property end-to-end (clearly TEST-labeled property name)
# --------------------------------------------------------------------------- #


def test_real_target_requires_full_coverage_then_unlabeled_works() -> None:
    db = Neo4jDatabase()
    ids = [_uid("t1"), _uid("t2")]
    try:
        db.initialize()
        _require_neo4j(db)
        _create_node(db, ids[0], "Src", risk_score=40.0,
                     risk_level="MEDIUM", target_days=3)
        _create_node(db, ids[1], "Dst", risk_score=60.0,
                     risk_level="HIGH", target_days=8)
        _create_relationship(db, ids[0], ids[1], "SUPPLIES")

        repository = GraphRepository()
        # A labeled build needs the property on EVERY node of the graph;
        # other/unrelated nodes in a shared dev DB will not have it.
        with pytest.raises(GraphDatasetValidationError):
            GraphDatasetBuilder(repository=repository).build(
                target_property=_TARGET_PROP,
            )

        dataset = GraphDatasetBuilder(repository=repository).build()
        assert dataset.num_nodes >= 2
        assert dataset.x.shape[0] == dataset.num_nodes
        m = dataset.node_id_to_index
        assert ids[0] in m and ids[1] in m
        assert list(dataset.edge_rel_types).count("SUPPLIES") >= 1
    finally:
        _delete_by_ids(db, ids)
        db.close()


# --------------------------------------------------------------------------- #
# 4. Builder is strictly read-only: node counts never change
# --------------------------------------------------------------------------- #


def test_builder_never_writes() -> None:
    db = Neo4jDatabase()
    probe = _uid("probe")
    try:
        db.initialize()
        _require_neo4j(db)
        _create_node(db, probe, "Probe", risk_score=50.0)

        def total() -> int:
            with db.get_session() as session:
                rec = session.run(
                    f"MATCH (n:`{_TEST_LABEL}`) RETURN count(n) AS c"
                ).single()
            return int(rec["c"]) if rec else 0

        before = total()
        GraphDatasetBuilder(repository=GraphRepository()).build()
        assert total() == before
    finally:
        _delete_by_ids(db, [probe])
        db.close()

