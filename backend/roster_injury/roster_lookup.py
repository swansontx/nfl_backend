"""Roster & Injury lookup service

Interface:
- get_player_status(game_id, player_id) -> str (one of 'ACT','RES','INA','DEV','CUT','RET')
"""

from typing import Dict

# placeholder local cache
_roster_cache: Dict[str, Dict[str, str]] = {}


def get_player_status(game_id: str, player_id: str) -> str:
    # TODO: load from outputs/game_rosters_YYYY.json and outputs/injury_game_index_YYYY.json
    return _roster_cache.get(game_id, {}).get(player_id, 'ACT')
