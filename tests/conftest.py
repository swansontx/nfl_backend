"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_game_id():
    """Provide a sample game_id for tests."""
    return "2025_10_KC_BUF"


@pytest.fixture
def sample_player_features():
    """Provide sample player features for tests."""
    return {
        'player_001': {
            'pass_yards': [250, 300, 280, 320, 290],
            'pass_tds': [2, 1, 3, 2, 2],
            'completions': [20, 25, 22, 28, 24],
            'attempts': [30, 35, 32, 38, 34]
        },
        'player_002': {
            'rush_yards': [80, 95, 70, 110, 85],
            'rush_tds': [1, 0, 1, 2, 0],
            'rush_attempts': [15, 18, 12, 20, 16]
        }
    }
