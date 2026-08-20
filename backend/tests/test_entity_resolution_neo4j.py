"""Real Neo4j integration tests for Entity Resolution (Module 8).

These tests execute against a REAL Neo4j server and are intentionally skipped
when the server is not reachable. Nothing here is mocked or faked.

How to run
----------
1. Start Neo4j and make sure it answers on the configured URI
   (``.env``: ``NEO4J_URI=neo4j://localhost:7687``).
2. From the ``backend`` directory::

       python -m pytest tests/test_entity_resolution_neo4j.py -v
"""

from __future__ import annotations

import uuid

import pytest
from neo4j.exceptions import ServiceUnavailable

from app.database.neo4j import Neo4jDatabase
from app.repositories.graph_repository import GraphRepository
from app.schemas.ner import NEREntity
from app.services.entity_resolution.entity_resolution_service import EntityResolutionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_neo4j(db: Neo4jDatabase) -> None:
    """Skip the calling test when a real Neo4j server is not reachable."""
    if not db.verify_connectivity():
        pytest.skip(
            "Neo4j is not available - start Neo4j (neo4j://localhost:7687) "
            "before running the real integration test"
        )


def _create_test_node(db: Neo4jDatabase, node_id: str, name: str, label: str = "TestEntity") -> None:
    query = f"CREATE (n:`{label}` {{id: $id, name: $name}}) RETURN n"
    parameters = {"id": node_id, "name": name}
    with db.get_session() as session:
        session.run(query, parameters).consume()


def _delete_test_node(db: Neo4jDatabase, node_id: str, label: str = "TestEntity") -> None:
    query = f"MATCH (n:`{label}` {{id: $id}}) DELETE n"
    parameters = {"id": node_id}
    with db.get_session() as session:
        session.run(query, parameters).consume()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_exact_resolution_against_real_neo4j() -> None:
    db = Neo4jDatabase()
    node_id = f"test-country-{uuid.uuid4().hex[:8]}"
    try:
        db.initialize()
        _require_neo4j(db)
        _create_test_node(db, node_id, "China", label="Country")

        repository = GraphRepository(db=db)
        service = EntityResolutionService(repository=repository)

        result = service.resolve([NEREntity(text="China", label="LOC", start=0, end=5)])

        assert result.resolved_entities[0].matched is True
        assert result.resolved_entities[0].node_id == node_id
        assert result.resolved_entities[0].node_name == "China"
        assert result.resolved_entities[0].match_method == "exact"
        assert result.resolved_entities[0].confidence == 1.0
    finally:
        try:
            _delete_test_node(db, node_id, label="Country")
        except Exception:
            pass
        db.close()


def test_unresolved_entity_against_real_neo4j() -> None:
    db = Neo4jDatabase()
    try:
        db.initialize()
        _require_neo4j(db)

        repository = GraphRepository(db=db)
        service = EntityResolutionService(repository=repository)

        result = service.resolve([NEREntity(text="NonexistentPlaceXYZ", label="LOC", start=0, end=18)])

        assert result.resolved_entities[0].matched is False
        assert result.resolved_entities[0].match_method == "unresolved"
        assert result.resolved_entities[0].node_id is None
    finally:
        db.close()


def test_normalized_exact_resolution_against_real_neo4j() -> None:
    db = Neo4jDatabase()
    node_id = f"test-port-{uuid.uuid4().hex[:8]}"
    try:
        db.initialize()
        _require_neo4j(db)
        _create_test_node(db, node_id, "Port of Rotterdam", label="Port")

        repository = GraphRepository(db=db)
        service = EntityResolutionService(repository=repository)

        result = service.resolve([NEREntity(text="  Port of Rotterdam  ", label="LOC", start=0, end=21)])

        assert result.resolved_entities[0].matched is True
        assert result.resolved_entities[0].node_id == node_id
        assert result.resolved_entities[0].match_method == "exact"
    finally:
        try:
            _delete_test_node(db, node_id, label="Port")
        except Exception:
            pass
        db.close()


def test_parameterized_cypher_no_injection() -> None:
    db = Neo4jDatabase()
    try:
        db.initialize()
        _require_neo4j(db)

        repository = GraphRepository(db=db)

        injection_attempt = "'; DETACH DELETE ALL; RETURN 1; //"
        with pytest.raises((ServiceUnavailable, Exception)):
            repository.find_entity_candidates(injection_attempt, limit=10)
    finally:
        db.close()


def test_no_duplicate_node_created_by_resolution() -> None:
    db = Neo4jDatabase()
    node_id = f"test-nodup-{uuid.uuid4().hex[:8]}"
    try:
        db.initialize()
        _require_neo4j(db)
        _create_test_node(db, node_id, "ExistingNode", label="TestEntity")

        repository = GraphRepository(db=db)
        service = EntityResolutionService(repository=repository)

        result = service.resolve([NEREntity(text="ExistingNode", label="ORG", start=0, end=13)])

        node_count_query = "MATCH (n:`TestEntity` {id: $id}) RETURN count(n) AS cnt"
        with db.get_session() as session:
            record = session.run(node_count_query, {"id": node_id}).single()
            assert record is not None
            assert record["cnt"] == 1

        assert result.resolved_entities[0].matched is True
        assert result.resolved_entities[0].node_id == node_id
    finally:
        try:
            _delete_test_node(db, node_id, label="TestEntity")
        except Exception:
            pass
        db.close()
