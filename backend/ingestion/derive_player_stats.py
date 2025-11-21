"""Derive player stats from play-by-play data.

Since nflverse hasn't published player_stats_2025.csv yet, we can
derive all player statistics from the play-by-play data which IS available.

This creates the same data that would be in player_stats.csv.

Usage:
    python -m backend.ingestion.derive_player_stats --year 2025
"""

from pathlib import Path
import argparse
import pandas as pd
from typing import Dict, List
from collections import defaultdict


def derive_player_stats(pbp_file: Path, output_file: Path, season: int) -> int:
    """Derive player stats from play-by-play data.

    Args:
        pbp_file: Path to play_by_play CSV
        output_file: Path to save derived stats
        season: NFL season year

    Returns:
        Number of player-week records created
    """
    print(f"\n{'='*60}")
    print(f"Deriving Player Stats from Play-by-Play")
    print(f"Season: {season}")
    print(f"{'='*60}\n")

    print(f"Loading play-by-play data from {pbp_file}...")
    df = pd.read_csv(pbp_file, low_memory=False)
    print(f"  Loaded {len(df):,} plays")

    # Filter to regular season and playoffs
    if 'season_type' in df.columns:
        df = df[df['season_type'].isin(['REG', 'POST'])]

    # Initialize stats containers
    # Key: (player_id, week)
    passing_stats = defaultdict(lambda: {
        'attempts': 0, 'completions': 0, 'yards': 0, 'tds': 0,
        'interceptions': 0, 'sacks': 0, 'sack_yards': 0, 'air_yards': 0,
        'player_name': '', 'team': '', 'position': 'QB'
    })

    rushing_stats = defaultdict(lambda: {
        'carries': 0, 'yards': 0, 'tds': 0, 'fumbles': 0,
        'first_downs': 0, 'player_name': '', 'team': '', 'position': ''
    })

    receiving_stats = defaultdict(lambda: {
        'targets': 0, 'receptions': 0, 'yards': 0, 'tds': 0,
        'air_yards': 0, 'yac': 0, 'first_downs': 0,
        'player_name': '', 'team': '', 'position': ''
    })

    # Process each play
    print("Processing plays...")

    for _, play in df.iterrows():
        week = play.get('week', 0)
        if pd.isna(week):
            continue
        week = int(week)

        # PASSING
        passer_id = play.get('passer_player_id')
        if pd.notna(passer_id) and passer_id:
            key = (passer_id, week)
            passing_stats[key]['player_name'] = play.get('passer_player_name', '')
            passing_stats[key]['team'] = play.get('posteam', '')

            # Pass attempt
            if play.get('pass_attempt', 0) == 1:
                passing_stats[key]['attempts'] += 1

                # Completion
                if play.get('complete_pass', 0) == 1:
                    passing_stats[key]['completions'] += 1
                    passing_stats[key]['yards'] += play.get('passing_yards', 0) or 0

                # TD
                if play.get('pass_touchdown', 0) == 1:
                    passing_stats[key]['tds'] += 1

                # INT
                if play.get('interception', 0) == 1:
                    passing_stats[key]['interceptions'] += 1

                # Air yards
                air = play.get('air_yards', 0)
                if pd.notna(air):
                    passing_stats[key]['air_yards'] += air

            # Sack
            if play.get('sack', 0) == 1:
                passing_stats[key]['sacks'] += 1
                passing_stats[key]['sack_yards'] += abs(play.get('yards_gained', 0) or 0)

        # RUSHING
        rusher_id = play.get('rusher_player_id')
        if pd.notna(rusher_id) and rusher_id:
            key = (rusher_id, week)
            rushing_stats[key]['player_name'] = play.get('rusher_player_name', '')
            rushing_stats[key]['team'] = play.get('posteam', '')

            if play.get('rush_attempt', 0) == 1:
                rushing_stats[key]['carries'] += 1
                rushing_stats[key]['yards'] += play.get('rushing_yards', 0) or 0

                if play.get('rush_touchdown', 0) == 1:
                    rushing_stats[key]['tds'] += 1

                if play.get('first_down_rush', 0) == 1:
                    rushing_stats[key]['first_downs'] += 1

                if play.get('fumble_lost', 0) == 1:
                    rushing_stats[key]['fumbles'] += 1

        # RECEIVING
        receiver_id = play.get('receiver_player_id')
        if pd.notna(receiver_id) and receiver_id:
            key = (receiver_id, week)
            receiving_stats[key]['player_name'] = play.get('receiver_player_name', '')
            receiving_stats[key]['team'] = play.get('posteam', '')

            # Target
            if play.get('pass_attempt', 0) == 1:
                receiving_stats[key]['targets'] += 1

                # Reception
                if play.get('complete_pass', 0) == 1:
                    receiving_stats[key]['receptions'] += 1
                    receiving_stats[key]['yards'] += play.get('receiving_yards', 0) or 0

                    # YAC
                    yac = play.get('yards_after_catch', 0)
                    if pd.notna(yac):
                        receiving_stats[key]['yac'] += yac

                if play.get('pass_touchdown', 0) == 1:
                    receiving_stats[key]['tds'] += 1

                if play.get('first_down_pass', 0) == 1:
                    receiving_stats[key]['first_downs'] += 1

                # Air yards
                air = play.get('air_yards', 0)
                if pd.notna(air):
                    receiving_stats[key]['air_yards'] += air

    # Combine into player records
    print("Aggregating player stats...")

    records = []

    # Process passers
    for (player_id, week), stats in passing_stats.items():
        if stats['attempts'] > 0 or stats['sacks'] > 0:
            records.append({
                'player_id': player_id,
                'player_name': stats['player_name'],
                'recent_team': stats['team'],
                'position': 'QB',
                'season': season,
                'week': week,
                'completions': stats['completions'],
                'attempts': stats['attempts'],
                'passing_yards': stats['yards'],
                'passing_tds': stats['tds'],
                'interceptions': stats['interceptions'],
                'sacks': stats['sacks'],
                'sack_yards': stats['sack_yards'],
                'passing_air_yards': stats['air_yards'],
                'carries': 0,
                'rushing_yards': 0,
                'rushing_tds': 0,
                'targets': 0,
                'receptions': 0,
                'receiving_yards': 0,
                'receiving_tds': 0,
                'receiving_air_yards': 0,
                'receiving_yards_after_catch': 0,
            })

    # Process rushers (merge with existing QB records if applicable)
    player_week_lookup = {(r['player_id'], r['week']): r for r in records}

    for (player_id, week), stats in rushing_stats.items():
        if stats['carries'] > 0:
            key = (player_id, week)
            if key in player_week_lookup:
                # Add to existing record (QB rushing)
                player_week_lookup[key]['carries'] = stats['carries']
                player_week_lookup[key]['rushing_yards'] = stats['yards']
                player_week_lookup[key]['rushing_tds'] = stats['tds']
            else:
                # New record for RB/other
                records.append({
                    'player_id': player_id,
                    'player_name': stats['player_name'],
                    'recent_team': stats['team'],
                    'position': 'RB',
                    'season': season,
                    'week': week,
                    'completions': 0,
                    'attempts': 0,
                    'passing_yards': 0,
                    'passing_tds': 0,
                    'interceptions': 0,
                    'sacks': 0,
                    'sack_yards': 0,
                    'passing_air_yards': 0,
                    'carries': stats['carries'],
                    'rushing_yards': stats['yards'],
                    'rushing_tds': stats['tds'],
                    'targets': 0,
                    'receptions': 0,
                    'receiving_yards': 0,
                    'receiving_tds': 0,
                    'receiving_air_yards': 0,
                    'receiving_yards_after_catch': 0,
                })

    # Rebuild lookup
    player_week_lookup = {(r['player_id'], r['week']): r for r in records}

    # Process receivers
    for (player_id, week), stats in receiving_stats.items():
        if stats['targets'] > 0:
            key = (player_id, week)
            if key in player_week_lookup:
                # Add to existing record
                player_week_lookup[key]['targets'] = stats['targets']
                player_week_lookup[key]['receptions'] = stats['receptions']
                player_week_lookup[key]['receiving_yards'] = stats['yards']
                player_week_lookup[key]['receiving_tds'] = stats['tds']
                player_week_lookup[key]['receiving_air_yards'] = stats['air_yards']
                player_week_lookup[key]['receiving_yards_after_catch'] = stats['yac']
            else:
                # New record for WR/TE
                records.append({
                    'player_id': player_id,
                    'player_name': stats['player_name'],
                    'recent_team': stats['team'],
                    'position': 'WR',  # Could be WR or TE
                    'season': season,
                    'week': week,
                    'completions': 0,
                    'attempts': 0,
                    'passing_yards': 0,
                    'passing_tds': 0,
                    'interceptions': 0,
                    'sacks': 0,
                    'sack_yards': 0,
                    'passing_air_yards': 0,
                    'carries': 0,
                    'rushing_yards': 0,
                    'rushing_tds': 0,
                    'targets': stats['targets'],
                    'receptions': stats['receptions'],
                    'receiving_yards': stats['yards'],
                    'receiving_tds': stats['tds'],
                    'receiving_air_yards': stats['air_yards'],
                    'receiving_yards_after_catch': stats['yac'],
                })

    # Calculate fantasy points
    for record in records:
        # Standard scoring
        record['fantasy_points'] = (
            record['passing_yards'] * 0.04 +
            record['passing_tds'] * 4 +
            record['interceptions'] * -2 +
            record['rushing_yards'] * 0.1 +
            record['rushing_tds'] * 6 +
            record['receiving_yards'] * 0.1 +
            record['receiving_tds'] * 6
        )

        # PPR scoring
        record['fantasy_points_ppr'] = (
            record['fantasy_points'] +
            record['receptions'] * 1
        )

    # Save to CSV
    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(['week', 'fantasy_points_ppr'], ascending=[True, False])
    result_df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"Derived Player Stats Complete")
    print(f"{'='*60}")
    print(f"  Total player-week records: {len(records):,}")
    print(f"  Unique players: {len(set(r['player_id'] for r in records)):,}")
    print(f"  Weeks covered: {sorted(set(r['week'] for r in records))}")
    print(f"  Output: {output_file}")
    print(f"{'='*60}\n")

    return len(records)


def main():
    parser = argparse.ArgumentParser(
        description='Derive player stats from play-by-play data'
    )
    parser.add_argument(
        '--year', type=int, default=2025,
        help='NFL season year (default: 2025)'
    )
    parser.add_argument(
        '--input-dir', type=Path, default=Path('inputs'),
        help='Directory containing play-by-play CSV'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('inputs'),
        help='Directory to save derived stats'
    )

    args = parser.parse_args()

    pbp_file = args.input_dir / f'play_by_play_{args.year}.csv'
    if not pbp_file.exists():
        print(f"ERROR: Play-by-play file not found: {pbp_file}")
        print("Run /fetch/nflverse first to download play-by-play data")
        exit(1)

    output_file = args.output_dir / f'player_stats_{args.year}.csv'

    derive_player_stats(pbp_file, output_file, args.year)


if __name__ == '__main__':
    main()
