"""Player name mapping utilities

Functions to map book/external player names to nflverse canonical player ids.

TODOs:
- Implement fuzzy matching, nickname handling, and team context
- Provide a small CLI to test mappings against a player lookup JSON
"""

from typing import Dict


def map_player_name(name: str, team: str, lookup: Dict[str, Dict]) -> str:
    """Return player_id or raise KeyError if not found"""
    # placeholder: naive exact match
    for pid,info in lookup.items():
        if info.get('name','').lower()==name.lower() and info.get('team','')==team:
            return pid
    raise KeyError(f'player not found: {name} ({team})')
