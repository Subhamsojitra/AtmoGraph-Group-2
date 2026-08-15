"""Integration tests for Neo4j database connection.

These tests require a running Neo4j instance at the configured URI.
They will be skipped if Neo4j is not available.
"""

import pytest
from neo4j.exceptions import ServiceUnavailable

from app.database.neo4j import Neo4jDatabase


def test_neo4j_integration():
    """Test real Neo4j connection when Neo4j is available."""
    # Create a Neo4jDatabase instance
    db = Neo4jDatabase()

    # Initialize the driver
    db.initialize()

    # Verify connectivity
    if not db.verify_connectivity():
        pytest.skip("Neo4j server is not available")

    # Test that we can create a session and run a query
    with db.get_session() as session:
        result = session.run("RETURN 1 AS result")
        record = result.single()
        assert record is not None
        assert record["result"] == 1

    # Test async initialization and connectivity (optional, but good to cover)
    # We'll test the async methods in a separate test if needed, but for now
    # we rely on the sync methods for the integration test.

    # Clean up
    db.close()