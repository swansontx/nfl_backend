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

    # Load roster data from nflverse weekly_rosters CSV
    rosters_file = rosters_source / f'weekly_rosters_{year}.csv'

    if not rosters_file.exists() or rosters_file.stat().st_size == 0:
        print(f"⚠ Rosters file not found or empty: {rosters_file}")
        print(f"⚠ Creating empty roster index")

        # Create empty placeholder
        roster_index = {}
        output_file = output_dir / f"game_rosters_{year}.json"
        with open(output_file, 'w') as f:
            json.dump(roster_index, f, indent=2)

        return roster_index

    print(f"Loading roster data from {rosters_file}")

    # Build roster index from weekly_rosters CSV
    # Format: season,team,position,depth_chart_position,jersey_number,status,
    #         player_id,player_name,week,game_type,...

    roster_index = {}
    player_count = 0

    try:
        import csv

        with open(rosters_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                season = row.get('season', '')
                team = row.get('team', '')
                week = row.get('week', '')
                player_id = row.get('gsis_id', row.get('player_id', ''))
                status = row.get('status', 'ACT')

                if not all([season, team, week, player_id]):
                    continue

                # Normalize status codes
                # nflverse uses: ACT, RES, NON, etc.
                # We use: ACT, RES, INA, DEV, CUT, RET
                status_map = {
                    'ACT': 'ACT',     # Active
                    'RES': 'RES',     # Reserve/Injured
                    'NON': 'INA',     # Inactive
                    'PUP': 'RES',     # Physically unable to perform
                    'SUS': 'INA',     # Suspended
                    'EXE': 'RES',     # Exempt
                    'RET': 'RET',     # Retired
                }
                normalized_status = status_map.get(status, 'ACT')

                # Build game_id for each week's games
                # We don't have exact matchups from roster data, so we'll create
                # team-week entries that can be matched to games later
                #
                # Store as: {year}_{week}_{team} -> player_id -> status
                week_key = f"{season}_{int(week):02d}_{team}"

                if week_key not in roster_index:
                    roster_index[week_key] = {}

                roster_index[week_key][player_id] = normalized_status
                player_count += 1

        print(f"✓ Loaded {player_count:,} player-week records")
        print(f"✓ Built roster index for {len(roster_index)} team-weeks")

    except Exception as e:
        print(f"✗ Error loading rosters: {e}")
        import traceback
        traceback.print_exc()
        roster_index = {}

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
