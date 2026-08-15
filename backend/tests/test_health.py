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

from app.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    """GET /api/v1/health responds with HTTP 200."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_reports_healthy() -> None:
    """GET /api/v1/health indicates a healthy service."""
    response = client.get("/api/v1/health")
    assert response.json()["status"] == "healthy"