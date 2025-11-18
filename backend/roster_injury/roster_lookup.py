"""Roster & Injury lookup service

DEPRECATED: Use backend.roster_injury.RosterInjuryService instead.
This file maintained for backwards compatibility only.
"""

from typing import Dict
from backend.roster_injury import RosterInjuryService
from backend.database.models import PlayerStatus

# Initialize service
_service = RosterInjuryService(use_cache=True)


def get_player_status(game_id: str, player_id: str) -> str:
    """
    Get player status for a game (legacy interface)

    DEPRECATED: Use RosterInjuryService.get_player_status() instead

    Returns:
        Status string: 'ACT','Q','D','P','OUT','IR','DEV','CUT','RET'
    """
    status_obj = _service.get_player_status(player_id, game_id)
    return status_obj.status.value


def get_player_confidence(game_id: str, player_id: str) -> float:
    """
    Get player confidence multiplier for a game

    Returns:
        Confidence value 0.0 to 1.0
    """
    status_obj = _service.get_player_status(player_id, game_id)
    return status_obj.confidence
