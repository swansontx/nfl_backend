"""Roster & Injury lookup service

Interface:
- get_player_status(game_id, player_id) -> str (one of 'ACT','RES','INA','DEV','CUT','RET')
"""

from typing import Dict
from pathlib import Path
import json


# placeholder local cache
_roster_cache: Dict[str, Dict[str, str]] = {}


def get_player_status(game_id: str, player_id: str) -> str:
    """Get player status for a specific game.

    Args:
        game_id: Format {season}_{week}_{away}_{home} (e.g., "2025_10_KC_BUF")
        player_id: nflverse player_id

    Returns:
        Status code: 'ACT', 'RES', 'INA', 'DEV', 'CUT', 'RET'

    TODO: Implement loading from outputs/game_rosters_YYYY.json and
          outputs/injury_game_index_YYYY.json using game_id to extract season
    """
    # Extract season from game_id for loading correct year's data
    # Example: "2025_10_KC_BUF" -> season = 2025
    # Then load: outputs/game_rosters_2025.json
    # and: outputs/injury_game_index_2025.json

    return _roster_cache.get(game_id, {}).get(player_id, 'ACT')
