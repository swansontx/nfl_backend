"""Roster & Injury lookup service

Interface:
- get_player_status(game_id, player_id) -> str (one of 'ACT','RES','INA','DEV','CUT','RET')
- load_roster_data(year) -> Load roster and injury data for a season
- get_injury_status(game_id, player_id) -> str (injury status if exists)
"""

from typing import Dict, Optional
from pathlib import Path
import json


# In-memory caches for roster and injury data
_roster_cache: Dict[str, Dict[str, str]] = {}
_injury_cache: Dict[str, Dict[str, str]] = {}
_loaded_years: set = set()


def load_roster_data(year: int, data_dir: Path = Path('outputs')) -> bool:
    """Load roster and injury data for a season.

    Args:
        year: Season year to load
        data_dir: Directory containing roster/injury JSON files

    Returns:
        True if data was loaded successfully

    Loads:
        - outputs/game_rosters_{year}.json
        - outputs/injury_game_index_{year}.json

    Call this before using get_player_status() for a new season.
    """
    if year in _loaded_years:
        return True  # Already loaded

    roster_file = data_dir / f'game_rosters_{year}.json'
    injury_file = data_dir / f'injury_game_index_{year}.json'

    loaded_any = False

    # Load roster data
    if roster_file.exists():
        try:
            with open(roster_file) as f:
                roster_data = json.load(f)
                _roster_cache.update(roster_data)
                print(f"✓ Loaded roster data for {year}: {len(roster_data)} team-weeks")
                loaded_any = True
        except Exception as e:
            print(f"✗ Error loading roster data for {year}: {e}")
    else:
        print(f"⚠ Roster file not found: {roster_file}")

    # Load injury data
    if injury_file.exists():
        try:
            with open(injury_file) as f:
                injury_data = json.load(f)
                _injury_cache.update(injury_data)
                print(f"✓ Loaded injury data for {year}: {len(injury_data)} team-weeks")
                loaded_any = True
        except Exception as e:
            print(f"✗ Error loading injury data for {year}: {e}")
    else:
        print(f"⚠ Injury file not found: {injury_file}")

    if loaded_any:
        _loaded_years.add(year)

    return loaded_any


def get_player_status(game_id: str, player_id: str, data_dir: Path = Path('outputs')) -> str:
    """Get player roster status for a specific game.

    Args:
        game_id: Format {season}_{week}_{away}_{home} (e.g., "2025_10_KC_BUF")
        player_id: nflverse player_id (gsis_id)
        data_dir: Directory containing roster JSON files

    Returns:
        Status code: 'ACT', 'RES', 'INA', 'DEV', 'CUT', 'RET'

    Note: Auto-loads roster data if not already loaded for this season.
    """
    # Extract season from game_id
    try:
        parts = game_id.split('_')
        if len(parts) >= 2:
            year = int(parts[0])

            # Auto-load data if not loaded
            if year not in _loaded_years:
                load_roster_data(year, data_dir)

            # Try exact game_id match first
            if game_id in _roster_cache:
                status = _roster_cache[game_id].get(player_id)
                if status:
                    return status

            # Try team-week matches (format: {year}_{week}_{team})
            week = parts[1]

            # Try both home and away teams if game_id has 4 parts
            if len(parts) >= 4:
                away_team = parts[2]
                home_team = parts[3]

                # Check away team roster
                away_key = f"{year}_{week}_{away_team}"
                if away_key in _roster_cache:
                    status = _roster_cache[away_key].get(player_id)
                    if status:
                        return status

                # Check home team roster
                home_key = f"{year}_{week}_{home_team}"
                if home_key in _roster_cache:
                    status = _roster_cache[home_key].get(player_id)
                    if status:
                        return status

    except Exception as e:
        print(f"Error parsing game_id {game_id}: {e}")

    # Default: assume active if not found
    return 'ACT'


def get_injury_status(game_id: str, player_id: str, data_dir: Path = Path('outputs')) -> Optional[str]:
    """Get player injury status for a specific game.

    Args:
        game_id: Format {season}_{week}_{away}_{home}
        player_id: nflverse player_id
        data_dir: Directory containing injury JSON files

    Returns:
        Injury status string if player is injured, None otherwise
        Statuses: 'Questionable', 'Doubtful', 'Out', 'IR', 'PUP', etc.
    """
    # Extract season from game_id
    try:
        parts = game_id.split('_')
        if len(parts) >= 2:
            year = int(parts[0])

            # Auto-load data if not loaded
            if year not in _loaded_years:
                load_roster_data(year, data_dir)

            # Try exact game_id match
            if game_id in _injury_cache:
                return _injury_cache[game_id].get(player_id)

            # Try team-week matches
            week = parts[1]

            if len(parts) >= 4:
                away_team = parts[2]
                home_team = parts[3]

                # Check away team injuries
                away_key = f"{year}_{week}_{away_team}"
                if away_key in _injury_cache:
                    status = _injury_cache[away_key].get(player_id)
                    if status:
                        return status

                # Check home team injuries
                home_key = f"{year}_{week}_{home_team}"
                if home_key in _injury_cache:
                    status = _injury_cache[home_key].get(player_id)
                    if status:
                        return status

    except Exception as e:
        print(f"Error parsing game_id {game_id}: {e}")

    return None


def clear_cache():
    """Clear all cached roster and injury data."""
    global _roster_cache, _injury_cache, _loaded_years
    _roster_cache = {}
    _injury_cache = {}
    _loaded_years = set()
    print("✓ Cleared roster and injury caches")
