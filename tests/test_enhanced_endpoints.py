"""Tests for enhanced API endpoints (prop value finder, player insights)."""

import pytest
from fastapi.testclient import TestClient
from backend.api.app import app

client = TestClient(app)


class TestPropValueEndpoint:
    """Tests for /api/v1/props/value endpoint."""

    def test_find_prop_value_returns_structure(self):
        """Test prop value finder returns proper structure."""
        response = client.get("/api/v1/props/value")
        assert response.status_code == 200
        data = response.json()

        assert 'total_opportunities' in data
        assert 'best_values' in data
        assert 'filters_applied' in data
        assert isinstance(data['best_values'], list)

    def test_prop_value_with_filters(self):
        """Test prop value finder with custom filters."""
        response = client.get("/api/v1/props/value?min_edge=7.0&min_grade=A&limit=5")
        assert response.status_code == 200
        data = response.json()

        # Verify filters were applied
        assert data['filters_applied']['min_edge'] == 7.0
        assert data['filters_applied']['min_grade'] == 'A'

    def test_prop_value_result_structure(self):
        """Test each prop value result has proper structure."""
        response = client.get("/api/v1/props/value")
        data = response.json()

        if data['best_values']:
            prop = data['best_values'][0]

            # Verify required fields
            assert 'player_name' in prop
            assert 'prop_type' in prop
            assert 'sportsbook_line' in prop
            assert 'model_projection' in prop
            assert 'confidence_interval' in prop
            assert 'recommendation' in prop
            assert 'edge_over' in prop
            assert 'edge_under' in prop
            assert 'value_grade' in prop
            assert 'confidence' in prop
            assert 'suggested_stake_pct' in prop
            assert 'sportsbook' in prop
            assert 'odds' in prop

            # Verify recommendation is valid
            assert prop['recommendation'] in ['OVER', 'UNDER', 'PASS']

            # Verify grade is valid
            assert prop['value_grade'] in ['A+', 'A', 'B+', 'B', 'C', 'F']


class TestPlayerInsightsEndpoint:
    """Tests for /api/v1/players/{player_id}/insights endpoint."""

    def test_get_player_insights_returns_structure(self):
        """Test player insights endpoint returns proper structure."""
        response = client.get("/api/v1/players/player_001/insights")
        assert response.status_code == 200
        data = response.json()

        assert 'player_id' in data
        assert 'player_name' in data
        assert 'position' in data
        assert 'team' in data
        assert 'insights' in data
        assert 'season_avg' in data
        assert 'recent_performance' in data

        assert isinstance(data['insights'], list)
        assert isinstance(data['recent_performance'], list)

    def test_player_insight_structure(self):
        """Test each player insight has proper structure."""
        response = client.get("/api/v1/players/player_001/insights")
        data = response.json()

        if data['insights']:
            insight = data['insights'][0]

            assert 'insight_type' in insight
            assert 'title' in insight
            assert 'description' in insight
            assert 'confidence' in insight
            assert 'impact_level' in insight
            assert 'supporting_data' in insight

            # Verify confidence is in valid range
            assert 0 <= insight['confidence'] <= 1

            # Verify impact level is valid
            assert insight['impact_level'] in ['high', 'medium', 'low']


class TestPropsCompareEndpoint:
    """Tests for /api/v1/props/compare endpoint."""

    def test_compare_props_returns_structure(self):
        """Test props compare endpoint returns proper structure."""
        response = client.get("/api/v1/props/compare?player_ids=p1,p2,p3")
        assert response.status_code == 200
        data = response.json()

        assert 'prop_type' in data
        assert 'players_compared' in data
        assert 'comparisons' in data
        assert 'best_value' in data

        assert isinstance(data['comparisons'], list)

    def test_compare_props_with_prop_type(self):
        """Test props compare with specific prop type."""
        response = client.get("/api/v1/props/compare?player_ids=p1,p2&prop_type=rushing_yards")
        assert response.status_code == 200
        data = response.json()

        assert data['prop_type'] == 'rushing_yards'

    def test_compare_props_limit(self):
        """Test props compare limits to 5 players."""
        response = client.get("/api/v1/props/compare?player_ids=p1,p2,p3,p4,p5,p6,p7")
        assert response.status_code == 200
        data = response.json()

        # Should be limited to 5 players
        assert len(data['comparisons']) <= 5


class TestGamePropSheetEndpoint:
    """Tests for /api/v1/games/{game_id}/prop-sheet endpoint."""

    def test_get_game_prop_sheet_returns_structure(self):
        """Test game prop sheet endpoint returns proper structure."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/prop-sheet")
        assert response.status_code == 200
        data = response.json()

        assert 'game_id' in data
        assert 'total_props' in data
        assert 'high_value_props' in data
        assert 'categories' in data
        assert 'top_plays' in data

        assert isinstance(data['categories'], dict)
        assert isinstance(data['top_plays'], list)

    def test_prop_sheet_categories(self):
        """Test prop sheet includes expected categories."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/prop-sheet")
        data = response.json()

        categories = data['categories']
        assert 'passing' in categories
        assert 'rushing' in categories
        assert 'receiving' in categories
        assert 'scoring' in categories

    def test_prop_sheet_top_plays_structure(self):
        """Test top plays have proper structure."""
        response = client.get("/api/v1/games/2025_10_KC_BUF/prop-sheet")
        data = response.json()

        if data['top_plays']:
            play = data['top_plays'][0]

            assert 'player' in play
            assert 'prop' in play
            assert 'edge' in play
            assert 'grade' in play
