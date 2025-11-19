"""Canonical: map odds API events to nflverse game ids

Placeholder utility to show expected interface.

The nflverse game_id format is: {season}_{week}_{away}_{home}
- season: 4-digit NFL season year (e.g., 2025 for 2025-26 season)
- week: 2-digit week number with leading zero (e.g., 01, 10, 18)
- away: 2-3 letter away team abbreviation (e.g., KC, BUF, NE)
- home: 2-3 letter home team abbreviation (e.g., KC, BUF, NE)

Example: 2025_10_KC_BUF (Week 10 of 2025 season, Kansas City @ Buffalo)

Note: Season year remains constant even for games played in January/February.
For example, a playoff game in January 2026 is still part of the 2025 season.
"""

def map_event_to_game(event_json: dict, season: int = 2025) -> str:
    """Return canonical game_id in format {season}_{week}_{away}_{home}.

    Args:
        event_json: Event data from odds API containing teams, date, etc.
        season: NFL season year (default 2025)

    Returns:
        Canonical game_id (e.g., 2025_10_KC_BUF)

    Raises:
        ValueError: If event cannot be mapped to a valid game_id
    """
    # TODO: implement mapping heuristics from event_json
    # - Extract week from date/schedule lookup
    # - Map team names to nflverse abbreviations
    # - Determine home/away teams
    return f"{season}_10_KC_BUF"
