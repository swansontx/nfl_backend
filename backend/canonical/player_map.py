"""Player name mapping utilities

Map player names from various sources (odds APIs, news sources) to
canonical nflverse player_ids.

This is critical for joining data across sources, as different APIs and
websites may use different name formats:
- "Patrick Mahomes" vs "P. Mahomes" vs "Pat Mahomes II"
- "Travis Kelce" vs "T. Kelce"

Features:
- Fuzzy name matching using rapidfuzz
- Team/position context for disambiguation
- Handles name variations and common misspellings
- Cached mappings for performance
"""

from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json
import re

# Try to import rapidfuzz (faster) or fall back to fuzzywuzzy
try:
    from rapidfuzz import fuzz, process
    FUZZY_LIB = 'rapidfuzz'
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        FUZZY_LIB = 'fuzzywuzzy'
    except ImportError:
        FUZZY_LIB = None
        print("Warning: No fuzzy matching library found. Install rapidfuzz or fuzzywuzzy.")


# In-memory cache of player mappings
_player_lookup: Dict[str, Dict] = {}
_name_to_id: Dict[str, str] = {}
_variations_to_id: Dict[str, str] = {}
_loaded_year: int = 0


def load_player_lookup(year: int = 2025, lookup_dir: Path = Path('inputs')) -> bool:
    """Load nflverse player lookup table into memory.

    Args:
        year: Season year for player lookup
        lookup_dir: Directory containing player_lookup_YYYY.json

    Returns:
        True if loaded successfully, False otherwise
    """
    global _player_lookup, _name_to_id, _variations_to_id, _loaded_year

    # Skip if already loaded for this year
    if _loaded_year == year and _player_lookup:
        return True

    lookup_file = lookup_dir / f"player_lookup_{year}.json"

    if not lookup_file.exists():
        print(f"Warning: Player lookup file not found: {lookup_file}")
        return False

    try:
        with open(lookup_file, 'r') as f:
            _player_lookup = json.load(f)

        # Build reverse name->id mapping
        _name_to_id = {}
        _variations_to_id = {}

        for pid, info in _player_lookup.items():
            if not info:
                continue

            player_name = info.get('player_name', info.get('name', ''))
            if not player_name:
                continue

            # Primary name mapping
            normalized = player_name.lower().strip()
            _name_to_id[normalized] = pid

            # Add all variations
            for variation in build_name_variations(player_name):
                _variations_to_id[variation] = pid

            # Add team-specific key for disambiguation
            team = info.get('team', info.get('recent_team', ''))
            if team:
                team_key = f"{normalized}_{team.lower()}"
                _name_to_id[team_key] = pid

        _loaded_year = year
        print(f"Loaded {len(_player_lookup)} players from {lookup_file}")
        return True

    except Exception as e:
        print(f"Error loading player lookup: {e}")
        return False


def map_player_name_to_id(player_name: str,
                          team: Optional[str] = None,
                          position: Optional[str] = None,
                          threshold: int = 85) -> Optional[str]:
    """Map a player name to nflverse player_id.

    Args:
        player_name: Player name as it appears in source data
        team: Optional team abbreviation for disambiguation
        position: Optional position for disambiguation
        threshold: Minimum fuzzy match score (0-100)

    Returns:
        nflverse player_id or None if no match found

    Strategy:
    1. Try exact match (case-insensitive)
    2. Try with team context for disambiguation
    3. Try name variations
    4. Try fuzzy match with threshold
    5. Use team/position context to filter results
    """
    if not player_name:
        return None

    # Normalize name
    normalized = player_name.lower().strip()
    # Remove suffixes like Jr., II, III
    normalized = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv)$', '', normalized)

    # 1. Try exact match
    if normalized in _name_to_id:
        return _name_to_id[normalized]

    # 2. Try with team context
    if team:
        team_key = f"{normalized}_{team.lower()}"
        if team_key in _name_to_id:
            return _name_to_id[team_key]

    # 3. Try name variations
    if normalized in _variations_to_id:
        return _variations_to_id[normalized]

    # 4. Try fuzzy matching
    if FUZZY_LIB and _name_to_id:
        # Get top matches
        if FUZZY_LIB == 'rapidfuzz':
            matches = process.extract(normalized, _name_to_id.keys(), limit=10, score_cutoff=threshold)
        else:
            matches = process.extractBests(normalized, _name_to_id.keys(), limit=10)
            matches = [(m[0], m[1]) for m in matches if m[1] >= threshold]

        if not matches:
            return None

        # If team provided, filter by team
        if team:
            team_lower = team.lower()
            team_matches = []
            for match_name, score in matches:
                pid = _name_to_id[match_name]
                player_info = _player_lookup.get(pid, {})
                player_team = player_info.get('team', player_info.get('recent_team', '')).lower()
                if player_team == team_lower:
                    team_matches.append((match_name, score))

            if team_matches:
                matches = team_matches

        # If position provided, filter by position
        if position and matches:
            pos_lower = position.upper()
            pos_matches = []
            for match_name, score in matches:
                pid = _name_to_id[match_name]
                player_info = _player_lookup.get(pid, {})
                player_pos = player_info.get('position', '').upper()
                if player_pos == pos_lower:
                    pos_matches.append((match_name, score))

            if pos_matches:
                matches = pos_matches

        # Return best match
        if matches:
            best_match = max(matches, key=lambda x: x[1])
            return _name_to_id[best_match[0]]

    return None


def get_best_matches(player_name: str,
                     team: Optional[str] = None,
                     limit: int = 5) -> List[Tuple[str, int, str]]:
    """Get top fuzzy matches for a player name.

    Args:
        player_name: Player name to match
        team: Optional team for filtering
        limit: Max matches to return

    Returns:
        List of (player_name, score, player_id) tuples
    """
    if not FUZZY_LIB or not _name_to_id:
        return []

    normalized = player_name.lower().strip()

    if FUZZY_LIB == 'rapidfuzz':
        matches = process.extract(normalized, _name_to_id.keys(), limit=limit)
    else:
        matches = process.extractBests(normalized, _name_to_id.keys(), limit=limit)

    results = []
    for match_name, score, *_ in matches:
        pid = _name_to_id[match_name]
        player_info = _player_lookup.get(pid, {})

        # Filter by team if provided
        if team:
            player_team = player_info.get('team', player_info.get('recent_team', ''))
            if player_team.lower() != team.lower():
                continue

        results.append((match_name, int(score), pid))

    return results


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
