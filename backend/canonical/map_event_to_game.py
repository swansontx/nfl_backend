"""Canonical: map odds API events to nflverse game ids

DEPRECATED: Use backend.canonical.GameMapper instead.
This file maintained for backwards compatibility only.
"""
from backend.canonical import GameMapper
from backend.database.session import get_db
from backend.database.models import Game


def map_event_to_game(event_json: dict) -> str:
    """
    Return canonical game_id (e.g., week_10_BUF) or raise if not found.

    DEPRECATED: Use GameMapper class for production use.

    Args:
        event_json: Event data from odds API

    Returns:
        Canonical game_id

    Raises:
        ValueError: If game cannot be matched with sufficient confidence
    """
    # Load games from database
    with get_db() as session:
        games = session.query(Game).all()

    if not games:
        raise ValueError("No games in database - run ingestion first")

    # Use GameMapper
    mapper = GameMapper(games)
    result = mapper.map_event_to_game(event_json)

    if result.game_id and result.confidence_score >= 70:
        return result.game_id
    else:
        raise ValueError(
            f"Could not map event to game with sufficient confidence. "
            f"Score: {result.confidence_score}, Method: {result.method}"
        )
