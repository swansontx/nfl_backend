"""Game ID utility functions for parsing and validating nflverse game IDs.

The nflverse game_id format is: {season}_{week}_{away}_{home}
- season: 4-digit NFL season year (e.g., 2025)
- week: 2-digit week number with leading zero (e.g., 01, 10, 18)
- away: 2-3 letter away team abbreviation
- home: 2-3 letter home team abbreviation

Example: 2025_10_KC_BUF
"""

from typing import Dict


def parse_game_id(game_id: str) -> Dict[str, any]:
    """Parse game_id into components.

    Args:
        game_id: Format {season}_{week}_{away}_{home}
                 Example: "2025_10_KC_BUF"

    Returns:
        dict with keys: season (int), week (int), away_team (str), home_team (str)

    Raises:
        ValueError: If game_id format is invalid

    Example:
        >>> parse_game_id("2025_10_KC_BUF")
        {'season': 2025, 'week': 10, 'away_team': 'KC', 'home_team': 'BUF'}
    """
    parts = game_id.split('_')

    if len(parts) != 4:
        raise ValueError(
            f"Invalid game_id format: {game_id}. "
            f"Expected format: {{season}}_{{week}}_{{away}}_{{home}}"
        )

    try:
        season = int(parts[0])
        week = int(parts[1])
    except ValueError as e:
        raise ValueError(
            f"Invalid game_id: {game_id}. Season and week must be integers."
        ) from e

    if season < 1990 or season > 2100:
        raise ValueError(f"Invalid season {season} in game_id: {game_id}")

    if week < 1 or week > 25:  # Regular season (1-18) + playoffs (19-22) + Pro Bowl (23-25)
        raise ValueError(f"Invalid week {week} in game_id: {game_id}")

    return {
        'season': season,
        'week': week,
        'away_team': parts[2],
        'home_team': parts[3]
    }


def build_game_id(season: int, week: int, away_team: str, home_team: str) -> str:
    """Build a game_id from components.

    Args:
        season: NFL season year (e.g., 2025)
        week: Week number (1-25)
        away_team: Away team abbreviation (e.g., "KC")
        home_team: Home team abbreviation (e.g., "BUF")

    Returns:
        game_id in format {season}_{week:02d}_{away}_{home}

    Example:
        >>> build_game_id(2025, 10, "KC", "BUF")
        '2025_10_KC_BUF'
    """
    if season < 1990 or season > 2100:
        raise ValueError(f"Invalid season: {season}")

    if week < 1 or week > 25:
        raise ValueError(f"Invalid week: {week}")

    return f"{season}_{week:02d}_{away_team}_{home_team}"


def extract_season_from_game_id(game_id: str) -> int:
    """Extract just the season year from a game_id.

    Args:
        game_id: Format {season}_{week}_{away}_{home}

    Returns:
        Season year as integer

    Example:
        >>> extract_season_from_game_id("2025_10_KC_BUF")
        2025
    """
    return parse_game_id(game_id)['season']


def extract_week_from_game_id(game_id: str) -> int:
    """Extract just the week number from a game_id.

    Args:
        game_id: Format {season}_{week}_{away}_{home}

    Returns:
        Week number as integer

    Example:
        >>> extract_week_from_game_id("2025_10_KC_BUF")
        10
    """
    return parse_game_id(game_id)['week']
