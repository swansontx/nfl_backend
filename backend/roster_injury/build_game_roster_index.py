"""Build game roster index

Creates a lookup index mapping game_id -> player_id -> roster_status
for quick roster lookups during modeling and API requests.

Output: outputs/game_rosters_YYYY.json

TODOs:
- Implement roster data loading from nflverse or other sources
- Handle practice squad, IR, suspensions
- Add historical roster tracking
- Support weekly roster changes
"""

from pathlib import Path
import argparse
import json
from typing import Dict


def build_roster_index(year: int,
                      rosters_source: Path,
                      output_dir: Path) -> Dict[str, Dict[str, str]]:
    """Build roster index for a season.

    Args:
        year: NFL season year
        rosters_source: Path to roster data source
        output_dir: Output directory for roster index

    Returns:
        Dictionary mapping game_id -> player_id -> status

    Status codes:
        - 'ACT': Active roster
        - 'RES': Reserve/Injured
        - 'INA': Inactive
        - 'DEV': Practice squad
        - 'CUT': Cut/released
        - 'RET': Retired

    Output format:
        {
            "2025_01_KC_BUF": {
                "player_001": "ACT",
                "player_002": "ACT",
                "player_003": "INA",
                ...
            },
            "2025_01_DAL_NYG": {
                ...
            }
        }

    TODO: Implement loading from nflverse roster data or NFL API
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Load roster data
    # Options:
    # 1. nflverse weekly roster data
    # 2. ESPN/NFL.com roster scraping
    # 3. Sports data APIs

    # Placeholder: create sample roster index
    roster_index = {
        f"{year}_01_KC_BUF": {
            "sample_player_001": "ACT",
            "sample_player_002": "ACT",
            "sample_player_003": "INA"
        },
        f"{year}_01_DAL_NYG": {
            "sample_player_004": "ACT",
            "sample_player_005": "RES"
        }
    }

    # Write output
    output_file = output_dir / f"game_rosters_{year}.json"
    with open(output_file, 'w') as f:
        json.dump(roster_index, f, indent=2)

    print(f"Built roster index for {year} season")
    print(f"Total games: {len(roster_index)}")
    print(f"Written to {output_file}")

    return roster_index


def update_roster_for_game(game_id: str,
                           roster_updates: Dict[str, str],
                           roster_file: Path) -> None:
    """Update roster status for a specific game.

    Args:
        game_id: Game ID to update
        roster_updates: Dictionary of player_id -> new_status
        roster_file: Path to roster index JSON file

    This is useful for last-minute roster changes (injuries, scratches, etc.)
    """
    # Load existing roster index
    if roster_file.exists():
        with open(roster_file) as f:
            roster_index = json.load(f)
    else:
        roster_index = {}

    # Update for this game
    if game_id not in roster_index:
        roster_index[game_id] = {}

    roster_index[game_id].update(roster_updates)

    # Save back
    with open(roster_file, 'w') as f:
        json.dump(roster_index, f, indent=2)

    print(f"Updated roster for {game_id}")
    print(f"Updated {len(roster_updates)} players")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Build game roster index')
    p.add_argument('--year', type=int, default=2025,
                   help='NFL season year')
    p.add_argument('--source', type=Path,
                   default=Path('inputs/rosters'),
                   help='Path to roster data source')
    p.add_argument('--output', type=Path,
                   default=Path('outputs'),
                   help='Output directory')
    args = p.parse_args()

    roster_index = build_roster_index(args.year, args.source, args.output)
    print(f"\nRoster index complete with {len(roster_index)} games")
