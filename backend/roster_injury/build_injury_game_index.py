"""Build injury-game index

Creates a lookup index mapping game_id -> player_id -> injury_status
for quick injury lookups during modeling and API requests.

Combines with roster index to provide complete player availability picture.

Output: outputs/injury_game_index_YYYY.json

TODOs:
- Implement injury data loading from multiple sources
- Handle injury status updates (questionable -> out, etc.)
- Track injury history over time
- Add probability estimates for questionable players
"""

from pathlib import Path
import argparse
import json
from typing import Dict, Optional
from datetime import datetime


def load_injury_reports(injuries_dir: Path, year: int) -> Dict[str, Dict]:
    """Load all injury reports for a season.

    Args:
        injuries_dir: Directory containing injury JSON files
        year: Season year

    Returns:
        Dictionary mapping date -> injury data

    Expected files: injuries_YYYYMMDD_parsed.json
    """
    injury_data = {}

    # TODO: Load all injury files for the year
    # pattern = f"injuries_{year}*_parsed.json"
    # for injury_file in injuries_dir.glob(pattern):
    #     date = injury_file.stem.split('_')[1]
    #     with open(injury_file) as f:
    #         injury_data[date] = json.load(f)

    # Placeholder
    injury_data['20251119'] = [
        {
            'player_id': 'sample_player_001',
            'status': 'Questionable',
            'injury': 'Knee',
            'week': 12
        }
    ]

    return injury_data


def map_injuries_to_games(injury_data: Dict[str, Dict],
                         schedule: Dict[str, Dict],
                         year: int) -> Dict[str, Dict[str, str]]:
    """Map injury data to specific games.

    Args:
        injury_data: Injury reports by date
        schedule: Game schedule (game_id -> game info)
        year: Season year

    Returns:
        Dictionary mapping game_id -> player_id -> injury_status

    Status codes:
        - 'Probable': Likely to play (>75%)
        - 'Questionable': Uncertain (25-75%)
        - 'Doubtful': Unlikely (<25%)
        - 'Out': Will not play
        - 'IR': Injured reserve
        - 'PUP': Physically unable to perform
    """
    injury_index = {}

    # TODO: Implement game-injury mapping logic
    # For each game:
    # 1. Determine game date from game_id/schedule
    # 2. Find injury report closest to game date
    # 3. Map player injuries to game_id

    # Placeholder
    injury_index[f'{year}_12_KC_BUF'] = {
        'sample_player_001': 'Questionable',
        'sample_player_002': 'Out'
    }

    return injury_index


def build_injury_index(year: int,
                      injuries_dir: Path,
                      schedule_path: Optional[Path],
                      output_dir: Path) -> Dict[str, Dict[str, str]]:
    """Build injury-game index for a season.

    Args:
        year: NFL season year
        injuries_dir: Directory containing injury reports
        schedule_path: Optional path to game schedule
        output_dir: Output directory for injury index

    Returns:
        Dictionary mapping game_id -> player_id -> injury_status

    Output format:
        {
            "2025_12_KC_BUF": {
                "player_001": "Questionable",
                "player_002": "Out",
                "player_003": "Doubtful"
            },
            "2025_12_DAL_NYG": {
                ...
            }
        }
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load injury reports
    injury_data = load_injury_reports(injuries_dir, year)
    print(f"Loaded injury data for {len(injury_data)} dates")

    # Load schedule (if provided)
    schedule = {}
    if schedule_path and schedule_path.exists():
        with open(schedule_path) as f:
            schedule = json.load(f)
    else:
        print("No schedule provided, using placeholder")

    # Map injuries to games
    injury_index = map_injuries_to_games(injury_data, schedule, year)

    # Write output
    output_file = output_dir / f"injury_game_index_{year}.json"
    with open(output_file, 'w') as f:
        json.dump(injury_index, f, indent=2)

    print(f"Built injury index for {year} season")
    print(f"Total games with injuries tracked: {len(injury_index)}")
    print(f"Written to {output_file}")

    return injury_index


def update_injury_for_game(game_id: str,
                          injury_updates: Dict[str, str],
                          injury_file: Path) -> None:
    """Update injury status for a specific game.

    Args:
        game_id: Game ID to update
        injury_updates: Dictionary of player_id -> injury_status
        injury_file: Path to injury index JSON file

    This is useful for last-minute injury updates (gameday decisions, etc.)
    """
    # Load existing injury index
    if injury_file.exists():
        with open(injury_file) as f:
            injury_index = json.load(f)
    else:
        injury_index = {}

    # Update for this game
    if game_id not in injury_index:
        injury_index[game_id] = {}

    injury_index[game_id].update(injury_updates)

    # Save back
    with open(injury_file, 'w') as f:
        json.dump(injury_index, f, indent=2)

    print(f"Updated injuries for {game_id}")
    print(f"Updated {len(injury_updates)} players")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Build injury-game index')
    p.add_argument('--year', type=int, default=2025,
                   help='NFL season year')
    p.add_argument('--injuries-dir', type=Path,
                   default=Path('outputs'),
                   help='Directory containing injury report JSONs')
    p.add_argument('--schedule', type=Path,
                   default=None,
                   help='Optional path to game schedule JSON')
    p.add_argument('--output', type=Path,
                   default=Path('outputs'),
                   help='Output directory')
    args = p.parse_args()

    injury_index = build_injury_index(
        args.year,
        args.injuries_dir,
        args.schedule,
        args.output
    )
    print(f"\nInjury index complete with {len(injury_index)} games tracked")
