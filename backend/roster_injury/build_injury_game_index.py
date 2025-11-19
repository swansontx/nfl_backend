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

    # Load all injury files for the year
    # Pattern: injuries_YYYYMMDD_parsed.json or injuries_YYYYMMDD.json
    pattern = f"injuries_{year}*.json"

    injury_files = list(injuries_dir.glob(pattern))

    if not injury_files:
        print(f"⚠ No injury files found matching {pattern} in {injuries_dir}")
        print(f"  Injury index will be empty")
        return {}

    print(f"Found {len(injury_files)} injury report files")

    for injury_file in injury_files:
        try:
            # Extract date from filename: injuries_YYYYMMDD_parsed.json -> YYYYMMDD
            filename_parts = injury_file.stem.split('_')
            if len(filename_parts) >= 2:
                date = filename_parts[1]  # YYYYMMDD

                with open(injury_file) as f:
                    injury_report = json.load(f)

                    # Handle different injury report formats
                    if isinstance(injury_report, list):
                        injury_data[date] = injury_report
                    elif isinstance(injury_report, dict):
                        # Flatten dict format to list
                        injuries_list = []
                        for team, players in injury_report.items():
                            if isinstance(players, list):
                                injuries_list.extend(players)
                        injury_data[date] = injuries_list if injuries_list else injury_report

                print(f"  ✓ Loaded injury report for {date}")
        except Exception as e:
            print(f"  ✗ Error loading {injury_file}: {e}")

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

    # Map injuries to games
    # Strategy: For each injury report date, find games in that week
    # Map injuries to all games for that team in that week

    if not injury_data:
        print("⚠ No injury data available")
        return injury_index

    if not schedule:
        print("⚠ No schedule provided, using date-based mapping")

        # Without schedule, map by week estimate
        # Assume Sunday games, map injury reports to nearest week
        for date_str, injuries in injury_data.items():
            try:
                # Parse date: YYYYMMDD
                if len(date_str) != 8:
                    continue

                year_str = date_str[:4]
                month = int(date_str[4:6])
                day = int(date_str[6:8])

                # Estimate week number (rough approximation)
                # NFL season starts in September, week 1 typically
                if month == 9:
                    week = ((day - 1) // 7) + 1
                elif month == 10:
                    week = 5 + ((day - 1) // 7)
                elif month == 11:
                    week = 9 + ((day - 1) // 7)
                elif month == 12:
                    week = 13 + ((day - 1) // 7)
                elif month == 1:
                    week = 18  # Playoffs
                else:
                    continue

                week = min(week, 18)  # Cap at week 18

                # Process injuries
                for injury in injuries:
                    if isinstance(injury, dict):
                        player_id = injury.get('player_id', injury.get('gsis_id', ''))
                        team = injury.get('team', '')
                        status = injury.get('status', injury.get('game_status', 'Questionable'))

                        if not player_id or not team:
                            continue

                        # Create game_id for this team/week
                        # Format: {year}_{week}_{team}
                        # We don't know opponent, so we'll use team-week key
                        game_key = f"{year_str}_{week:02d}_{team}"

                        if game_key not in injury_index:
                            injury_index[game_key] = {}

                        injury_index[game_key][player_id] = status

            except Exception as e:
                print(f"  ✗ Error processing date {date_str}: {e}")

        print(f"✓ Mapped injuries to {len(injury_index)} team-week combinations")

    else:
        # With schedule, map injuries to actual games
        print("⚠ Schedule-based mapping not yet implemented")
        print("  Using date-based approximation instead")

        # Fall back to date-based mapping (same logic as above)
        for date_str, injuries in injury_data.items():
            try:
                if len(date_str) != 8:
                    continue

                year_str = date_str[:4]
                month = int(date_str[4:6])
                day = int(date_str[6:8])

                if month == 9:
                    week = ((day - 1) // 7) + 1
                elif month == 10:
                    week = 5 + ((day - 1) // 7)
                elif month == 11:
                    week = 9 + ((day - 1) // 7)
                elif month == 12:
                    week = 13 + ((day - 1) // 7)
                elif month == 1:
                    week = 18
                else:
                    continue

                week = min(week, 18)

                for injury in injuries:
                    if isinstance(injury, dict):
                        player_id = injury.get('player_id', injury.get('gsis_id', ''))
                        team = injury.get('team', '')
                        status = injury.get('status', injury.get('game_status', 'Questionable'))

                        if not player_id or not team:
                            continue

                        game_key = f"{year_str}_{week:02d}_{team}"

                        if game_key not in injury_index:
                            injury_index[game_key] = {}

                        injury_index[game_key][player_id] = status

            except Exception as e:
                print(f"  ✗ Error processing date {date_str}: {e}")

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
