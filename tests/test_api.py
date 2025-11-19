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
        data = response.json()
        assert data['status'] == 'ok'
        assert 'timestamp' in data


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


# ============================================================================
# Phase 1: News & Injuries Tests
# ============================================================================

class TestNewsEndpoint:
    """Tests for /api/v1/news endpoint."""

    def test_get_news_returns_list(self):
        """Test news endpoint returns list of news items."""
        response = client.get("/api/v1/news")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_news_with_limit(self):
        """Test news endpoint respects limit parameter."""
        response = client.get("/api/v1/news?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5

    def test_get_news_with_category_filter(self):
        """Test news endpoint filters by category."""
        response = client.get("/api/v1/news?category=injury")
        assert response.status_code == 200
        data = response.json()
        # All returned items should be injury category (or empty if none)
        if data:
            assert all(item['category'] == 'injury' for item in data)

    def test_get_news_with_team_filter(self):
        """Test news endpoint filters by team."""
        response = client.get("/api/v1/news?team=KC")
        assert response.status_code == 200
        # Should not error even if no KC news


class TestGameInjuriesEndpoint:
    """Tests for /api/v1/games/{game_id}/injuries endpoint."""

    def test_get_game_injuries_returns_structure(self):
        """Test game injuries endpoint returns proper structure."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/injuries")
        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert 'game_id' in data
        assert 'away_team' in data
        assert 'home_team' in data
        assert 'away_injuries' in data
        assert 'home_injuries' in data
        assert 'last_updated' in data

        assert isinstance(data['away_injuries'], list)
        assert isinstance(data['home_injuries'], list)

    def test_get_game_injuries_parses_game_id(self):
        """Test that game_id is properly parsed."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/injuries")
        assert response.status_code == 200
        data = response.json()
        assert data['away_team'] == 'KC'
        assert data['home_team'] == 'BUF'


# ============================================================================
# Phase 2: Insights & Narrative Tests
# ============================================================================

class TestGameInsightsEndpoint:
    """Tests for /api/v1/games/{game_id}/insights endpoint."""

    def test_get_game_insights_returns_list(self):
        """Test insights endpoint returns list of insights."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/insights")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_insight_structure(self):
        """Test each insight has proper structure."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/insights")
        data = response.json()

        if data:  # If insights are returned
            insight = data[0]
            assert 'insight_type' in insight
            assert 'title' in insight
            assert 'description' in insight
            assert 'confidence' in insight
            assert 'supporting_data' in insight
            assert 0 <= insight['confidence'] <= 1


class TestGameNarrativeEndpoint:
    """Tests for /api/v1/games/{game_id}/narrative endpoint."""

    def test_get_game_narrative_returns_list(self):
        """Test narrative endpoint returns list of narratives."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/narrative")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_narrative_structure(self):
        """Test each narrative has proper structure."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/narrative")
        data = response.json()

        if data:
            narrative = data[0]
            assert 'narrative_type' in narrative
            assert 'content' in narrative
            assert 'generated_at' in narrative


# ============================================================================
# Phase 3: Content Tests
# ============================================================================

class TestGameContentEndpoint:
    """Tests for /api/v1/games/{game_id}/content endpoint."""

    def test_get_game_content_returns_list(self):
        """Test content endpoint returns list of content items."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/content")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_content_with_type_filter(self):
        """Test content endpoint filters by type."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/content?content_type=video")
        assert response.status_code == 200
        data = response.json()

        if data:
            assert all(item['content_type'] == 'video' for item in data)

    def test_content_with_limit(self):
        """Test content endpoint respects limit."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/content?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 2

    def test_content_structure(self):
        """Test each content item has proper structure."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/content")
        data = response.json()

        if data:
            item = data[0]
            assert 'content_type' in item
            assert 'title' in item
            assert 'source' in item
            assert 'url' in item
            assert 'published_at' in item


# ============================================================================
# Bonus: Weather Tests
# ============================================================================

class TestGameWeatherEndpoint:
    """Tests for /api/v1/games/{game_id}/weather endpoint."""

    def test_get_game_weather_returns_data(self):
        """Test weather endpoint returns weather data."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/weather")
        assert response.status_code == 200
        data = response.json()

        # Verify all required fields
        assert 'temperature' in data
        assert 'temp_unit' in data
        assert 'condition' in data
        assert 'wind_speed' in data
        assert 'wind_unit' in data
        assert 'humidity' in data
        assert 'precipitation_chance' in data
        assert 'is_dome' in data

    def test_weather_data_types(self):
        """Test weather data has correct types."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/weather")
        data = response.json()

        assert isinstance(data['temperature'], int)
        assert isinstance(data['wind_speed'], int)
        assert isinstance(data['humidity'], int)
        assert isinstance(data['precipitation_chance'], int)
        assert isinstance(data['is_dome'], bool)
