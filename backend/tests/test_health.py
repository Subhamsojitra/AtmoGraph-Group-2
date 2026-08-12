"""Tests for the health check endpoint (Module 1)."""

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