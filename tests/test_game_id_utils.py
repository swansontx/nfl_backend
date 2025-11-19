"""Tests for game_id utility functions."""

import pytest
from backend.canonical.game_id_utils import (
    parse_game_id,
    build_game_id,
    extract_season_from_game_id,
    extract_week_from_game_id
)


class TestParseGameId:
    """Tests for parse_game_id function."""

    def test_parse_valid_game_id(self):
        """Test parsing a valid game_id."""
        result = parse_game_id("2025_10_KC_BUF")
        assert result['season'] == 2025
        assert result['week'] == 10
        assert result['away_team'] == 'KC'
        assert result['home_team'] == 'BUF'

    def test_parse_week_with_leading_zero(self):
        """Test parsing week with leading zero."""
        result = parse_game_id("2025_01_DAL_NYG")
        assert result['week'] == 1
        assert result['away_team'] == 'DAL'
        assert result['home_team'] == 'NYG'

    def test_parse_playoff_game(self):
        """Test parsing playoff game (week > 18)."""
        result = parse_game_id("2025_20_KC_BUF")
        assert result['season'] == 2025
        assert result['week'] == 20

    def test_invalid_format_missing_parts(self):
        """Test error on invalid format with missing parts."""
        with pytest.raises(ValueError, match="Invalid game_id format"):
            parse_game_id("2025_10_KC")

    def test_invalid_format_extra_parts(self):
        """Test error on invalid format with extra parts."""
        with pytest.raises(ValueError, match="Invalid game_id format"):
            parse_game_id("2025_10_KC_BUF_extra")

    def test_invalid_season_not_integer(self):
        """Test error when season is not an integer."""
        with pytest.raises(ValueError, match="Season and week must be integers"):
            parse_game_id("ABCD_10_KC_BUF")

    def test_invalid_week_not_integer(self):
        """Test error when week is not an integer."""
        with pytest.raises(ValueError, match="Season and week must be integers"):
            parse_game_id("2025_AB_KC_BUF")

    def test_invalid_season_out_of_range(self):
        """Test error when season is out of valid range."""
        with pytest.raises(ValueError, match="Invalid season"):
            parse_game_id("1899_10_KC_BUF")

    def test_invalid_week_out_of_range(self):
        """Test error when week is out of valid range."""
        with pytest.raises(ValueError, match="Invalid week"):
            parse_game_id("2025_99_KC_BUF")


class TestBuildGameId:
    """Tests for build_game_id function."""

    def test_build_valid_game_id(self):
        """Test building a valid game_id."""
        result = build_game_id(2025, 10, "KC", "BUF")
        assert result == "2025_10_KC_BUF"

    def test_build_with_leading_zero(self):
        """Test week formatting with leading zero."""
        result = build_game_id(2025, 1, "DAL", "NYG")
        assert result == "2025_01_DAL_NYG"

    def test_build_playoff_game(self):
        """Test building playoff game_id."""
        result = build_game_id(2025, 20, "KC", "BUF")
        assert result == "2025_20_KC_BUF"

    def test_invalid_season(self):
        """Test error with invalid season."""
        with pytest.raises(ValueError, match="Invalid season"):
            build_game_id(1899, 10, "KC", "BUF")

    def test_invalid_week(self):
        """Test error with invalid week."""
        with pytest.raises(ValueError, match="Invalid week"):
            build_game_id(2025, 99, "KC", "BUF")


class TestExtractSeasonFromGameId:
    """Tests for extract_season_from_game_id function."""

    def test_extract_season(self):
        """Test extracting season from game_id."""
        season = extract_season_from_game_id("2025_10_KC_BUF")
        assert season == 2025


class TestExtractWeekFromGameId:
    """Tests for extract_week_from_game_id function."""

    def test_extract_week(self):
        """Test extracting week from game_id."""
        week = extract_week_from_game_id("2025_10_KC_BUF")
        assert week == 10

    def test_extract_week_with_leading_zero(self):
        """Test extracting week with leading zero."""
        week = extract_week_from_game_id("2025_01_DAL_NYG")
        assert week == 1
