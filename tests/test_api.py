"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.api.app import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self):
        """Test that health endpoint returns OK status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {'status': 'ok'}


class TestRecomputeEndpoint:
    """Tests for /admin/recompute endpoint."""

    def test_recompute_with_valid_game_id(self):
        """Test recompute endpoint with valid game_id."""
        response = client.post(
            "/admin/recompute",
            json={"game_id": "2025_10_KC_BUF"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'started'
        assert data['game_id'] == '2025_10_KC_BUF'

    def test_recompute_without_game_id(self):
        """Test recompute endpoint without game_id."""
        response = client.post("/admin/recompute", json={})
        assert response.status_code == 422  # Validation error


class TestProjectionsEndpoint:
    """Tests for /game/{game_id}/projections endpoint."""

    def test_get_projections_returns_empty_list(self):
        """Test projections endpoint returns structure (placeholder)."""
        response = client.get("/game/2025_10_KC_BUF/projections")
        assert response.status_code == 200
        data = response.json()
        assert 'game_id' in data
        assert 'projections' in data
        assert data['game_id'] == '2025_10_KC_BUF'
        assert isinstance(data['projections'], list)
