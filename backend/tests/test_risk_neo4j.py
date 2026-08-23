"""Real Neo4j integration tests for the risk state update module (Module 9).

These tests execute against a REAL Neo4j server and are intentionally skipped
when the server is not reachable. Nothing here is mocked or faked. Run from the
``backend`` directory:

    python -m pytest tests/test_risk_neo4j.py -v

When Neo4j is down the tests are reported as ``skipped`` (never as passed) with
a clear message, so the suite never requires a live database.

Tests verify: existing node lookup, risk state read, risk state update,
persistence, invalid risk state rejection, nonexistent node handling,
parameterized Cypher (no injection), and no duplicate node creation.
"""

from __future__ import annotations

import uuid

import pytest
from neo4j.exceptions import ServiceUnavailable

from app.database.neo4j import Neo4jDatabase
from app.repositories.graph_repository import GraphRepository
from app.schemas.risk import RiskLevelUpdateRequest
from app.services.risk.exceptions import EntityNotFoundError
from app.services.risk.risk_service import RiskService

_TEST_LABEL = "TestRisk"


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


def _read_risk_state(db: Neo4jDatabase, node_id: str) -> dict[str, object]:
    """Read the persisted risk state of a node, or an empty dict."""
    query = f"MATCH (n:`{_TEST_LABEL}` {{id: $id}}) RETURN n AS n"
    with db.get_session() as session:
        record = session.run(query, {"id": node_id}).single()
    if record is None:
        return {}
    node = record["n"]
    return {
        "name": node.get("name"),
        "risk_score": node.get("risk_score"),
        "risk_level": node.get("risk_level"),
    }


def _node_count(db: Neo4jDatabase, node_id: str) -> int:
    """Count nodes with the given id (used to assert no duplicates)."""
    query = f"MATCH (n:`{_TEST_LABEL}` {{id: $id}}) RETURN count(n) AS cnt"
    with db.get_session() as session:
        record = session.run(query, {"id": node_id}).single()
    return int(record["cnt"]) if record else 0


def _delete_test_node(db: Neo4jDatabase, node_id: str) -> None:
    """Delete a test node by id, ignoring any cleanup error."""
    query = f"MATCH (n:`{_TEST_LABEL}` {{id: $id}}) DELETE n"
    with db.get_session() as session:
        session.run(query, {"id": node_id}).consume()
# --------------------------------------------------------------------------- #
# 1/2/3. Existing node lookup + risk state read + update + persistence
# --------------------------------------------------------------------------- #


def test_risk_state_update_persists() -> None:
    db = Neo4jDatabase()
    node_id = f"test-risk-{uuid.uuid4().hex[:8]}"
    try:
        db.initialize()
        _require_neo4j(db)
        _create_test_node(db, node_id, "Port of Rotterdam", risk_score=30.0, risk_level="LOW")

        repository = GraphRepository(db=db)
        service = RiskService(repository=repository)

        result = service.update_risk_state(
            RiskLevelUpdateRequest(
                entity_id=node_id,
                entity_name="Port of Rotterdam",
                risk_score=85.0,
                reason="Port strike",
            )
        )

        # Response reports the previous -> new transition.
        assert result.updated is True
        assert result.entity_id == node_id
        assert result.previous_risk_score == 30.0
        assert result.new_risk_score == 85.0
        assert result.previous_risk_level == "LOW"
        assert result.new_risk_level == "HIGH"

        # The write actually persisted in Neo4j.
        state = _read_risk_state(db, node_id)
        assert state["risk_score"] == 85.0
        assert state["risk_level"] == "HIGH"
        assert state["name"] == "Port of Rotterdam"
    finally:
        try:
            _delete_test_node(db, node_id)
        except Exception:
            pass
        db.close()


# --------------------------------------------------------------------------- #
# 4. Invalid risk state -> rejected, nothing changed
# --------------------------------------------------------------------------- #


def test_invalid_risk_state_is_rejected() -> None:
    db = Neo4jDatabase()
    node_id = f"test-risk-{uuid.uuid4().hex[:8]}"
    try:
        db.initialize()
        _require_neo4j(db)
        _create_test_node(db, node_id, "Port of Rotterdam", risk_score=30.0, risk_level="LOW")

        repository = GraphRepository(db=db)
        service = RiskService(repository=repository)

        # Bypass Pydantic so the service's own validation is exercised.
        invalid = RiskLevelUpdateRequest.model_construct(
            entity_id=node_id, risk_score=101.0
        )
        with pytest.raises(ValueError, match="risk_score must be between"):
            service.update_risk_state(invalid)

        # Rejected write must not have changed the persisted state.
        state = _read_risk_state(db, node_id)
        assert state["risk_score"] == 30.0
        assert state["risk_level"] == "LOW"
    finally:
        try:
            _delete_test_node(db, node_id)
        except Exception:
            pass
        db.close()


# --------------------------------------------------------------------------- #
# 5. Nonexistent node -> EntityNotFoundError, node is NOT created
# --------------------------------------------------------------------------- #


def test_nonexistent_node_is_not_created() -> None:
    db = Neo4jDatabase()
    missing_id = f"test-missing-{uuid.uuid4().hex[:8]}"
    try:
        db.initialize()
        _require_neo4j(db)

        repository = GraphRepository(db=db)
        service = RiskService(repository=repository)

        with pytest.raises(EntityNotFoundError):
            service.update_risk_state(
                RiskLevelUpdateRequest(entity_id=missing_id, risk_score=85.0)
            )

        # Module 9 must never create a new node.
        assert _node_count(db, missing_id) == 0
    finally:
        try:
            _delete_test_node(db, missing_id)
        except Exception:
            pass
        db.close()
# --------------------------------------------------------------------------- #
# 6. No duplicate node creation
# --------------------------------------------------------------------------- #


def test_repeated_updates_do_not_create_duplicates() -> None:
    db = Neo4jDatabase()
    node_id = f"test-risk-{uuid.uuid4().hex[:8]}"
    try:
        db.initialize()
        _require_neo4j(db)
        _create_test_node(db, node_id, "Port of Rotterdam", risk_score=30.0, risk_level="LOW")

        repository = GraphRepository(db=db)
        service = RiskService(repository=repository)

        service.update_risk_state(RiskLevelUpdateRequest(entity_id=node_id, risk_score=50.0))
        service.update_risk_state(RiskLevelUpdateRequest(entity_id=node_id, risk_score=90.0))

        # Exactly one node exists after repeated updates.
        assert _node_count(db, node_id) == 1
        state = _read_risk_state(db, node_id)
        assert state["risk_score"] == 90.0
        assert state["risk_level"] == "HIGH"
    finally:
        try:
            _delete_test_node(db, node_id)
        except Exception:
            pass
        db.close()


# --------------------------------------------------------------------------- #
# 7. Parameterized Cypher (injection attempt cannot create nodes or corrupt data)
# --------------------------------------------------------------------------- #


def test_parameterized_cypher_no_injection() -> None:
    db = Neo4jDatabase()
    safe_id = f"test-risk-{uuid.uuid4().hex[:8]}"
    injection_id = "'; DETACH DELETE ALL; RETURN 1; //"
    try:
        db.initialize()
        _require_neo4j(db)
        _create_test_node(db, safe_id, "Port of Rotterdam", risk_score=30.0, risk_level="LOW")

        repository = GraphRepository(db=db)
        service = RiskService(repository=repository)

        with pytest.raises(EntityNotFoundError):
            service.update_risk_state(
                RiskLevelUpdateRequest(entity_id=injection_id, risk_score=85.0)
            )

        # Parameterization means the injection string was treated as data: no
        # node with that id was created and the safe node is untouched.
        assert _node_count(db, injection_id) == 0
        state = _read_risk_state(db, safe_id)
        assert state["risk_score"] == 30.0
        assert state["risk_level"] == "LOW"
    finally:
        try:
            _delete_test_node(db, safe_id)
        except Exception:
            pass
        db.close()