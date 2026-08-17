"""Real Neo4j integration tests (Module 3).

These tests execute against a REAL Neo4j server and are intentionally skipped
when the server is not reachable. Nothing here is mocked or faked.

How to run the real integration test
------------------------------------
1. Start Neo4j and make sure it answers on the configured URI
   (``.env``: ``NEO4J_URI=neo4j://localhost:7687``).
2. From the ``backend`` directory::

       python -m pytest tests/test_neo4j_integration.py -v

When Neo4j is down the tests are reported as ```skipped`` (never as passed)
with a clear message. The driver is always closed in a ``finally`` block so
resources are released even when a step fails.
"""

import pytest

from app.database.neo4j import Neo4jDatabase


def _require_neo4j(db: Neo4jDatabase) -> None:
    """Skip the calling test when a real Neo4j server is not reachable.

    The connectivity check performs an actual ``RETURN 1`` round-trip; a
    skipped test is never reported as passed.
    """
    if not db.verify_connectivity():
        pytest.skip(
            "Neo4j is not available - start Neo4j (neo4j://localhost:7687) "
            "before running the real integration test"
        )


def test_neo4j_driver_initializes() -> None:
    """The shared manager builds a reusable driver from settings."""
    db = Neo4jDatabase()
    try:
        db.initialize()
        assert db.is_initialized is True
        _require_neo4j(db)
    finally:
        db.close()
    assert db.is_initialized is False


def test_neo4j_connectivity_and_basic_query() -> None:
    """Real connection: session executes ``RETURN 1 AS result`` and returns 1.

    Verifies, against the real server when available:
    * driver initializes and connects,
    * a session can be opened and closed,
    * ``RETURN 1 AS result`` returns exactly ``1``,
    * the driver and session are closed afterwards.
    """
    db = Neo4jDatabase()
    try:
        db.initialize()
        assert db.is_initialized is True
        _require_neo4j(db)

        with db.get_session() as session:
            result = session.run("RETURN 1 AS result")
            record = result.single()
            assert record is not None, "query returned no record"
            assert record["result"] == 1

        # Session was closed by the context manager; the driver is still
        # usable, so another session can be opened.
        with db.get_session() as session:
            second = session.run("RETURN 1 AS result").single()
            assert second["result"] == 1
    finally:
        db.close()
    assert db.is_initialized is False