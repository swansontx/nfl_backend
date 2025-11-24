"""Player name mapping utilities

Map player names from various sources (odds APIs, news sources) to
canonical nflverse player_ids.

This is critical for joining data across sources, as different APIs and
websites may use different name formats:
- "Patrick Mahomes" vs "P. Mahomes" vs "Pat Mahomes II"
- "Travis Kelce" vs "T. Kelce"
"""

from typing import Optional, Dict, List
from pathlib import Path
import json

from backend.nfl_calendar import get_current_nfl_season

# Try to import fuzzywuzzy for fuzzy matching
try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


# In-memory cache of player mappings
_player_lookup: Dict[str, str] = {}
_name_to_id: Dict[str, str] = {}


def load_player_lookup(year: int = None, lookup_dir: Path = Path('inputs')) -> bool:
    """Load nflverse player lookup table into memory.

    Args:
        year: Season year for player lookup
        lookup_dir: Directory containing player_lookup_YYYY.json

    Returns:
        True if loaded successfully, False otherwise

    Expected JSON format:
        {
            "player_id": {
                "name": "Full Name",
                "team": "BUF",
                "position": "QB",
                ...
            }
        }
    """
    global _player_lookup, _name_to_id
    year = year or get_current_nfl_season()

    lookup_file = lookup_dir / f"player_lookup_{year}.json"

    if lookup_file.exists():
        with open(lookup_file) as f:
            _player_lookup = json.load(f)

            # Build reverse name->id mapping with variations
            _name_to_id.clear()
            for pid, info in _player_lookup.items():
                if 'name' in info:
                    name = info['name']
                    # Add full name
                    _name_to_id[name.lower()] = pid

                    # Add common variations
                    for variation in build_name_variations(name):
                        if variation not in _name_to_id:
                            _name_to_id[variation] = pid

                # Also map by display_name if available
                if 'display_name' in info:
                    _name_to_id[info['display_name'].lower()] = pid

        print(f"Loaded {len(_player_lookup)} players from {lookup_file}")
        return True
    else:
        print(f"Player lookup file not found: {lookup_file}")
        return False


def map_player_name_to_id(player_name: str,
                          team: Optional[str] = None,
                          position: Optional[str] = None,
                          fuzzy_threshold: int = 85) -> Optional[str]:
    """Map a player name to nflverse player_id.

    Args:
        player_name: Player name as it appears in source data
        team: Optional team abbreviation for disambiguation
        position: Optional position for disambiguation
        fuzzy_threshold: Minimum score for fuzzy match (0-100)

    Returns:
        nflverse player_id or None if no match found

    Matching strategy:
    1. Try exact match (case-insensitive)
    2. Try name variations
    3. Try fuzzy match with threshold
    4. Use team/position context to disambiguate
    """
    if not _name_to_id:
        # Auto-load if not loaded
        load_player_lookup()

    # Normalize name
    normalized = player_name.lower().strip()

    # 1. Try exact match
    if normalized in _name_to_id:
        return _name_to_id[normalized]

    # 2. Try generated variations
    for variation in build_name_variations(player_name):
        if variation in _name_to_id:
            return _name_to_id[variation]

    # 3. Fuzzy matching (if available)
    if FUZZY_AVAILABLE and _name_to_id:
        matches = process.extractBests(normalized, list(_name_to_id.keys()), limit=5, score_cutoff=fuzzy_threshold)

        if matches:
            # Filter by team/position if provided
            for match_name, score in matches:
                pid = _name_to_id[match_name]
                player_info = _player_lookup.get(pid, {})

                # Check team match if provided
                if team and player_info.get('team'):
                    if player_info['team'].upper() != team.upper():
                        continue

                # Check position match if provided
                if position and player_info.get('position'):
                    if player_info['position'].upper() != position.upper():
                        continue

                return pid

            # If no filter match, return best match
            if matches:
                return _name_to_id[matches[0][0]]

    return None


def get_player_info(player_id: str) -> Optional[Dict]:
    """Get full player information from player_id.

    Args:
        player_id: nflverse player_id

    Returns:
        Player info dict or None if not found
    """
    return _player_lookup.get(player_id)


def build_name_variations(full_name: str) -> List[str]:
    """Generate common name variations for matching.

    Args:
        full_name: Full player name (e.g., "Patrick Mahomes")

    Returns:
        List of name variations

    Examples:
        >>> build_name_variations("Patrick Mahomes")
        ['patrick mahomes', 'p. mahomes', 'p mahomes', 'mahomes']
    """
    variations = []
    normalized = full_name.lower().strip()
    variations.append(normalized)

    parts = normalized.split()
    if len(parts) >= 2:
        first, *middle, last = parts

        # First initial + last name
        variations.append(f"{first[0]}. {last}")
        variations.append(f"{first[0]} {last}")

        # Last name only
        variations.append(last)

        # First + last (skip middle)
        if middle:
            variations.append(f"{first} {last}")

    return variations


if __name__ == '__main__':
    # Example usage
    print("Player name mapping utilities")
    print("\nExample name variations:")
    print(build_name_variations("Patrick Lavon Mahomes II"))
    print(build_name_variations("Travis Kelce"))
