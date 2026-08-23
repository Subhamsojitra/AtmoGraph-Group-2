"""Real Neo4j integration tests for the risk propagation module (Module 10).

These tests execute against a REAL Neo4j server and are intentionally skipped
when the server is not reachable. Nothing here is mocked or faked. Run from the
``backend`` directory:

    python -m pytest tests/test_risk_propagation_neo4j.py -v

When Neo4j is down the tests are reported as ``skipped`` (never as passed) with
a clear message, so the suite never requires a live database.

Tests verify: downstream traversal, relationship-type filtering, risk
decay/severity calculation, source-node lookup, cycle prevention, no unintended
node creation, no duplicate records, and parameterized Cypher.
"""

from __future__ import annotations

import uuid

import pytest

from app.database.neo4j import Neo4jDatabase
from app.repositories.graph_repository import GraphRepository
from app.schemas.risk_propagation import RiskPropagationRequest
from app.services.risk.exceptions import EntityNotFoundError
from app.services.risk.risk_service import RiskService
from app.services.risk_propagation.risk_propagation_service import RiskPropagationService

_TEST_LABEL = "TestPropagation"


def _require_neo4j(db: Neo4jDatabase) -> None:
    """Skip the calling test when a real Neo4j server is not reachable."""
    if not db.verify_connectivity():
        pytest.skip(
            "Neo4j is not available - start Neo4j (neo4j://localhost:7687) "
            "before running the real integration test"
        )


def _create_test_node(
    db: Neo4jDatabase,
    node_id: str,
    name: str,
    risk_score: float | None = None,
    risk_level: str | None = None,
) -> None:
    """Create a test node (optionally with an initial risk state)."""
    score_clause = ", n.risk_score = $risk_score" if risk_score is not None else ""
    level_clause = ", n.risk_level = $risk_level" if risk_level is not None else ""
    query = (
        f"CREATE (n:`{_TEST_LABEL}` {{id: $id, name: $name"
        f"{score_clause}{level_clause}}}) RETURN n"
    )
    parameters = {"id": node_id, "name": name}
    if risk_score is not None:
        parameters["risk_score"] = risk_score
    if risk_level is not None:
        parameters["risk_level"] = risk_level
    with db.get_session() as session:
        session.run(query, parameters).consume()


def _create_test_relationship(
    db: Neo4jDatabase,
    source_id: str,
    target_id: str,
    rel_type: str = "SUPPLIES",
) -> None:
    """Create a directed relationship from source to target."""
    query = (
        f"MATCH (a:`{_TEST_LABEL}` {{id: $source}}) "
        f"MATCH (b:`{_TEST_LABEL}` {{id: $target}}) "
        f"CREATE (a)-[:`{rel_type}`]->(b)"
    )
    with db.get_session() as session:
        session.run(query, {"source": source_id, "target": target_id}).consume()


def _node_count(db: Neo4jDatabase, node_id: str) -> int:
    """Count nodes with the given id (used to assert no duplicates/creations)."""
    query = f"MATCH (n:`{_TEST_LABEL}` {{id: $id}}) RETURN count(n) AS cnt"
    with db.get_session() as session:
        record = session.run(query, {"id": node_id}).single()
    return int(record["cnt"]) if record else 0


def _delete_test_node(db: Neo4jDatabase, node_id: str) -> None:
    """Delete a test node by id (cascading), ignoring any cleanup error."""
    query = f"MATCH (n:`{_TEST_LABEL}` {{id: $id}}) DETACH DELETE n"
    with db.get_session() as session:
        session.run(query, {"id": node_id}).consume()


def _service(db: Neo4jDatabase) -> RiskPropagationService:
    """Build a real propagation service bound to the passed database."""
    repository = GraphRepository(db=db)
    risk_service = RiskService(repository=repository)
    return RiskPropagationService(
        repository=repository,
        risk_service=risk_service,
    )


# --------------------------------------------------------------------------- #
# 1. Downstream traversal + propagated severity + persistence
# --------------------------------------------------------------------------- #


def test_downstream_traversal_and_propagated_risk() -> None:
    db = Neo4jDatabase()
    source = f"test-prop-src-{uuid.uuid4().hex[:8]}"
    first = f"test-prop-fst-{uuid.uuid4().hex[:8]}"
    second = f"test-prop-snd-{uuid.uuid4().hex[:8]}"
    node_ids = [source, first, second]
    try:
        db.initialize()
        _require_neo4j(db)
        _create_test_node(db, source, "Port of Rotterdam", risk_score=80.0, risk_level="HIGH")
        _create_test_node(db, first, "Gigafactory Assembly")
        _create_test_node(db, second, "Regional Warehouse Hub")
        _create_test_relationship(db, source, first, "SUPPLIES")
        _create_test_relationship(db, first, second, "SUPPLIES")

        service = _service(db)
        response = service.propagate(
            RiskPropagationRequest(
                entity_id=source,
                entity_name="Port of Rotterdam",
                risk_score=80.0,
                max_depth=2,
                attenuation=0.5,
            )
        )

        assert response.propagated is True
        assert response.source_entity_id == source
        assert response.source_risk_score == 80.0
        assert response.affected_count == 2

        by_id = {e.entity_id: e for e in response.affected_entities}
        assert first in by_id
        assert second in by_id

        # Depth-1 at attenuation 0.5: 80 * 0.5 = 40 -> MEDIUM.
        assert by_id[first].depth == 1
        assert by_id[first].propagated_risk_score == 40.0
        assert by_id[first].propagated_risk_level == "MEDIUM"

        # Depth-2: 80 * 0.25 = 20 -> LOW.
        assert by_id[second].depth == 2
        assert by_id[second].propagated_risk_score == 20.0
        assert by_id[second].propagated_risk_level == "LOW"

        # The source entity itself is never reported as affected.
        assert all(e.entity_id != source for e in response.affected_entities)
    finally:
        for node_id in node_ids:
            try:
                _delete_test_node(db, node_id)
            except Exception:
                pass
        db.close()


# --------------------------------------------------------------------------- #
# 2. Relationship-type filtering
# --------------------------------------------------------------------------- #


def test_relationship_type_filtering() -> None:
    db = Neo4jDatabase()
    source = f"test-prop-src-{uuid.uuid4().hex[:8]}"
    rel_allowed = f"test-prop-all-{uuid.uuid4().hex[:8]}"
    rel_filtered = f"test-prop-fil-{uuid.uuid4().hex[:8]}"
    node_ids = [source, rel_allowed, rel_filtered]
    try:
        db.initialize()
        _require_neo4j(db)
        _create_test_node(db, source, "Port of Rotterdam", risk_score=60.0)
        _create_test_node(db, rel_allowed, "Allowed Supplier")
        _create_test_node(db, rel_filtered, "Filtered Counterparty")
        _create_test_relationship(db, source, rel_allowed, "SUPPLIES")
        _create_test_relationship(db, source, rel_filtered, "TRANSPORTS")

        service = _service(db)
        response = service.propagate(
            RiskPropagationRequest(
                entity_id=source,
                risk_score=60.0,
                max_depth=1,
                relationship_types=["SUPPLIES"],
            )
        )

        affected_ids = {e.entity_id for e in response.affected_entities}
        assert rel_allowed in affected_ids
        assert rel_filtered not in affected_ids
    finally:
        for node_id in node_ids:
            try:
                _delete_test_node(db, node_id)
            except Exception:
                pass
        db.close()


# --------------------------------------------------------------------------- #
# 3. Cycle prevention: repeated traversal yields one record per node
# --------------------------------------------------------------------------- #

def test_cycle_prevention_deduplicates_nodes() -> None:
    db = Neo4jDatabase()
    a = f"test-prop-a-{uuid.uuid4().hex[:8]}"
    b = f"test-prop-b-{uuid.uuid4().hex[:8]}"
    node_ids = [a, b]
    try:
        db.initialize()
        _require_neo4j(db)
        _create_test_node(db, a, "Supplier A", risk_score=80.0)
        _create_test_node(db, b, "Supplier B")
        _create_test_relationship(db, a, b, "SUPPLIES")
        _create_test_relationship(db, b, a, "SUPPLIES")  # cycle back to A

        service = _service(db)
        response = service.propagate(
            RiskPropagationRequest(entity_id=a, risk_score=80.0, max_depth=4)
        )

        # B is reached once; the reverse edge back to A never re-adds A.
        assert response.affected_count == 1
        assert response.affected_entities[0].entity_id == b
        assert _node_count(db, a) == 1
        assert _node_count(db, b) == 1
    finally:
        for node_id in node_ids:
            try:
                _delete_test_node(db, node_id)
            except Exception:
                pass
        db.close()


# --------------------------------------------------------------------------- #
# 4. Nonexistent source -> EntityNotFoundError, node NOT created
# --------------------------------------------------------------------------- #

def test_nonexistent_source_is_not_created() -> None:
    db = Neo4jDatabase()
    missing_id = f"test-prop-missing-{uuid.uuid4().hex[:8]}"
    try:
        db.initialize()
        _require_neo4j(db)

        service = _service(db)
        with pytest.raises(EntityNotFoundError):
            service.propagate(
                RiskPropagationRequest(entity_id=missing_id, risk_score=85.0)
            )

        # Module 10 must never create a node.
        assert _node_count(db, missing_id) == 0
    finally:
        try:
            _delete_test_node(db, missing_id)
        except Exception:
            pass
        db.close()


# --------------------------------------------------------------------------- #
# 5. Parameterized Cypher (injection attempt cannot create nodes or corrupt data)
# --------------------------------------------------------------------------- #

def test_parameterized_cypher_no_injection() -> None:
    db = Neo4jDatabase()
    safe_id = f"test-prop-safe-{uuid.uuid4().hex[:8]}"
    injection_id = "'; DETACH DELETE ALL; RETURN 1; //"
    try:
        db.initialize()
        _require_neo4j(db)
        _create_test_node(db, safe_id, "Safe Source", risk_score=30.0, risk_level="LOW")

        service = _service(db)
        with pytest.raises(EntityNotFoundError):
            service.propagate(
                RiskPropagationRequest(entity_id=injection_id, risk_score=85.0)
            )

        # Parameterization means the injection string was treated as data: the
        # safe node is untouched and no node with the injection id was created.
        assert _node_count(db, safe_id) == 1
        assert _node_count(db, injection_id) == 0
    finally:
        try:
            _delete_test_node(db, safe_id)
        except Exception:
            pass
        db.close()

