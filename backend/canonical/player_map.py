"""Player name mapping utilities

Map player names from various sources (odds APIs, news sources) to
canonical nflverse player_ids.

This is critical for joining data across sources, as different APIs and
websites may use different name formats:
- "Patrick Mahomes" vs "P. Mahomes" vs "Pat Mahomes II"
- "Travis Kelce" vs "T. Kelce"

TODOs:
- Implement fuzzy name matching (e.g., using fuzzywuzzy)
- Load nflverse player lookup tables
- Handle name variations and common misspellings
- Cache mappings for performance
- Add team context to improve matching accuracy
"""

from typing import Optional, Dict, List
from pathlib import Path
import json


# In-memory cache of player mappings
_player_lookup: Dict[str, str] = {}
_name_to_id: Dict[str, str] = {}


def load_player_lookup(year: int = 2025, lookup_dir: Path = Path('inputs')) -> None:
    """Load nflverse player lookup table into memory.

    Args:
        year: Season year for player lookup
        lookup_dir: Directory containing player_lookup_YYYY.json

    TODO: Implement loading from inputs/player_lookup_YYYY.json
    Expected format:
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

    lookup_file = lookup_dir / f"player_lookup_{year}.json"

    # TODO: Load actual data
    # if lookup_file.exists():
    #     with open(lookup_file) as f:
    #         _player_lookup = json.load(f)
    #         # Build reverse name->id mapping
    #         for pid, info in _player_lookup.items():
    #             _name_to_id[info['name'].lower()] = pid

    print(f"TODO: Load player lookup from {lookup_file}")


def map_player_name_to_id(player_name: str,
                          team: Optional[str] = None,
                          position: Optional[str] = None) -> Optional[str]:
    """Map a player name to nflverse player_id.

    Args:
        player_name: Player name as it appears in source data
        team: Optional team abbreviation for disambiguation
        position: Optional position for disambiguation

    Returns:
        nflverse player_id or None if no match found

    TODO: Implement fuzzy matching logic
    Strategy:
    1. Try exact match (case-insensitive)
    2. Try partial match (last name)
    3. Try fuzzy match with threshold
    4. Use team/position context to disambiguate
    """
    # Normalize name
    normalized = player_name.lower().strip()

    # Try exact match
    if normalized in _name_to_id:
        return _name_to_id[normalized]

    # TODO: Implement fuzzy matching
    # from fuzzywuzzy import process
    # matches = process.extractBests(normalized, _name_to_id.keys(), limit=5)
    # Filter by team/position if provided
    # Return best match above threshold

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
