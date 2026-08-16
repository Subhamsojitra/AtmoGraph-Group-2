"""Tests for the health check endpoint (Module 1).

These tests set the required Neo4j environment variables to allow the
application to import the configuration successfully.
"""
import os

# Set fake Neo4j environment variables before importing the app
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    """GET /api/v1/health responds with HTTP 200 while the API is up."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_reports_degraded_when_neo4j_unavailable() -> None:
    """GET /api/v1/health reports 'degraded'/'disconnected' when Neo4j is
    unreachable (as it is in this CI environment, where no real server runs)."""
    response = client.get("/api/v1/health")
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["service"] == "AtmoGraph API"
    assert payload["neo4j"] == "disconnected"


def test_health_reports_healthy_when_neo4j_connected(monkeypatch) -> None:
    """GET /api/v1/health reports 'healthy'/'connected' when the connectivity
    check succeeds.

    This is a unit-level test of the response contract: it patches only the
    manager's connectivity probe so the endpoint mapping can be verified
    without faking the real integration test's database connection.
    """
    from app.database.neo4j import neo4j_db

    monkeypatch.setattr(neo4j_db, "verify_connectivity", lambda: True)

    response = client.get("/api/v1/health")
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "AtmoGraph API"
    assert payload["neo4j"] == "connected"