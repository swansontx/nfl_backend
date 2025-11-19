"""Fetch NFL injury data from nflverse for enhanced DNP tracking.

nflverse provides weekly injury reports with detailed status information:
- Out: Player will not play
- Doubtful: Player unlikely to play
- Questionable: Player's availability uncertain
- Probable: Player likely to play (deprecated in recent years)

This data enhances our DNP tracking by providing actual injury reasons instead of
just inferring from play counts.

Data Source: https://github.com/nflverse/nflverse-data/releases/
File: injuries.csv or injuries.parquet

Output: JSON mapping (season, week, player_gsis_id) -> injury_status
"""

from pathlib import Path
import argparse
import requests
import json
from typing import Dict, List, Optional
import pandas as pd


def fetch_injury_data(
    season: int,
    output_dir: Path,
    cache_dir: Optional[Path] = None
) -> Dict:
    """Fetch injury data for a season from nflverse.

    Args:
        season: Season year (e.g., 2024)
        output_dir: Directory to save injury data JSON
        cache_dir: Optional cache directory

    Returns:
        Dict mapping (season, week, player_gsis_id) -> injury_info
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fetching injury data for {season} season")
    print(f"{'='*60}\n")

    # nflverse injury data URL
    url = f"https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"

    # Try parquet first, fallback to CSV
    cache_file = cache_dir / f"injuries_{season}.parquet" if cache_dir else None

    try:
        # Download injury data
        if cache_file and cache_file.exists():
            print(f"✓ Using cached injury data: {cache_file}")
            df = pd.read_parquet(cache_file)
        else:
            print(f"📥 Downloading injury data from nflverse...")
            df = pd.read_parquet(url)

            if cache_file:
                df.to_parquet(cache_file)
                print(f"✓ Cached to: {cache_file}")

        print(f"✓ Loaded {len(df)} injury records")

        # Process injury data into lookup dict
        injury_map = _process_injury_data(df, season)

        # Save as JSON
        output_file = output_dir / f"{season}_injuries.json"
        with open(output_file, 'w') as f:
            json.dump(injury_map, f, indent=2)

        print(f"✓ Saved injury data to: {output_file}")
        print(f"✓ Total player-weeks with injury data: {len(injury_map)}")

        return injury_map

    except Exception as e:
        print(f"⚠️  Failed to fetch injury data: {e}")
        print(f"⚠️  Continuing without injury data (will use is_active inference only)")

        # Return empty dict if fetch fails
        output_file = output_dir / f"{season}_injuries.json"
        output_file.write_text('{}')

        return {}


def _process_injury_data(df: pd.DataFrame, season: int) -> Dict:
    """Process injury DataFrame into lookup dictionary.

    Args:
        df: Injury DataFrame from nflverse
        season: Season year

    Returns:
        Dict with keys like "2024_1_00-0012345" -> injury_info
    """
    injury_map = {}

    # Expected columns in nflverse injuries:
    # - season (int)
    # - week (int)
    # - gsis_id (str) - player GSIS ID
    # - full_name (str)
    # - position (str)
    # - team (str)
    # - report_status (str) - Out, Doubtful, Questionable, Probable
    # - report_primary_injury (str) - Injury type (Knee, Ankle, etc.)
    # - practice_status (str) - Full, Limited, DNP

    for _, row in df.iterrows():
        # Create composite key: season_week_gsis_id
        week = row.get('week', 0)
        gsis_id = row.get('gsis_id', '')

        if not gsis_id:
            continue

        # Key format: "2024_1_00-0012345"
        key = f"{season}_{week}_{gsis_id}"

        injury_info = {
            'season': season,
            'week': int(week),
            'player_id': gsis_id,
            'full_name': row.get('full_name', ''),
            'position': row.get('position', ''),
            'team': row.get('team', ''),
            'report_status': row.get('report_status', ''),  # Out, Doubtful, Questionable
            'primary_injury': row.get('report_primary_injury', ''),  # Knee, Ankle, etc.
            'practice_status': row.get('practice_status', ''),  # Full, Limited, DNP
            'date_modified': row.get('date_modified', '')  # Last updated
        }

        # Determine DNP reason
        report_status = injury_info['report_status']
        if report_status == 'Out':
            injury_info['dnp_reason'] = 'injury_out'
        elif report_status == 'Doubtful':
            injury_info['dnp_reason'] = 'injury_doubtful'
        elif report_status == 'Questionable':
            injury_info['dnp_reason'] = 'injury_questionable'
        else:
            injury_info['dnp_reason'] = None

        # Add flag for expected absence
        injury_info['expected_to_play'] = report_status not in ['Out', 'Doubtful']

        injury_map[key] = injury_info

    return injury_map


def merge_injury_into_features(
    features_file: Path,
    injury_file: Path,
    output_file: Path
) -> Dict:
    """Merge injury data into player features.

    Args:
        features_file: Path to player features JSON
        injury_file: Path to injury data JSON
        output_file: Path to save merged features

    Returns:
        Updated features dict
    """
    print(f"\n{'='*60}")
    print(f"Merging injury data into features")
    print(f"{'='*60}\n")

    # Load features
    with open(features_file, 'r') as f:
        features = json.load(f)

    # Load injury data
    with open(injury_file, 'r') as f:
        injuries = json.load(f)

    print(f"✓ Loaded {len(features)} players")
    print(f"✓ Loaded {len(injuries)} injury records")

    # Merge injury data into features
    updated_count = 0
    matched_count = 0

    for player_id, games in features.items():
        for game in games:
            season = game.get('season', '')
            week = game.get('week', '')

            # Try to match injury data
            # Key format: "2024_1_00-0012345"
            injury_key = f"{season}_{week}_{player_id}"

            if injury_key in injuries:
                injury_info = injuries[injury_key]

                # Add injury data to game features
                game['injury_status'] = injury_info['report_status']
                game['injury_type'] = injury_info['primary_injury']
                game['practice_status'] = injury_info['practice_status']
                game['expected_to_play'] = injury_info['expected_to_play']

                # Update dnp_reason if player was inactive
                if not game.get('is_active', True):
                    game['dnp_reason'] = injury_info['dnp_reason']
                    updated_count += 1

                matched_count += 1

    # Save merged features
    with open(output_file, 'w') as f:
        json.dump(features, f, indent=2)

    print(f"✓ Matched {matched_count} player-week injury records")
    print(f"✓ Updated DNP reasons for {updated_count} inactive games")
    print(f"✓ Saved merged features to: {output_file}")

    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch and merge NFL injury data for enhanced DNP tracking'
    )
    parser.add_argument('--season', type=int, required=True,
                       help='Season year (e.g., 2024)')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('outputs/injuries'),
                       help='Output directory for injury data')
    parser.add_argument('--cache-dir', type=Path,
                       default=Path('cache'),
                       help='Cache directory')
    parser.add_argument('--features-file', type=Path,
                       help='Optional: Player features file to merge with')
    parser.add_argument('--merge-output', type=Path,
                       help='Optional: Output path for merged features')

    args = parser.parse_args()

    # Fetch injury data
    injury_map = fetch_injury_data(
        season=args.season,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )

    # If features file provided, merge injury data
    if args.features_file and args.merge_output:
        injury_file = args.output_dir / f"{args.season}_injuries.json"
        merge_injury_into_features(
            features_file=args.features_file,
            injury_file=injury_file,
            output_file=args.merge_output
        )
