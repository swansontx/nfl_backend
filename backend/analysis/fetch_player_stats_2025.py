"""Fetch and aggregate player stats from NFLverse for 2025 season.

This creates the training data we need for prop models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import gzip
import io


def download_player_stats(year: int):
    """Download player stats from NFLverse.

    URL format: https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.parquet
    """

    print(f"\n{'='*80}")
    print(f"FETCHING PLAYER STATS FOR {year}")
    print(f"{'='*80}\n")

    # Try parquet format first
    parquet_url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.parquet"

    print(f"Downloading from: {parquet_url}")

    try:
        response = requests.get(parquet_url, timeout=60)

        if response.status_code == 200:
            # Save temporarily
            temp_file = Path(f'inputs/player_stats_{year}.parquet')
            temp_file.write_bytes(response.content)

            # Read parquet
            stats_df = pd.read_parquet(temp_file)

            print(f"✅ Downloaded {len(stats_df)} player-game records")
            print()

            return stats_df

        else:
            print(f"❌ Failed to download: HTTP {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Try CSV format as fallback
    print("\nTrying CSV format...")
    csv_url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.csv"

    try:
        response = requests.get(csv_url, timeout=60)

        if response.status_code == 200:
            stats_df = pd.read_csv(io.StringIO(response.text))

            print(f"✅ Downloaded {len(stats_df)} player-game records")
            print()

            return stats_df

    except Exception as e:
        print(f"❌ Error: {e}")

    return pd.DataFrame()


def analyze_player_stats(stats_df):
    """Analyze what data we have."""

    print(f"\n{'='*80}")
    print(f"PLAYER STATS ANALYSIS")
    print(f"{'='*80}\n")

    print(f"Total records: {len(stats_df)}")
    print()

    # Check weeks available
    if 'week' in stats_df.columns:
        weeks = sorted(stats_df['week'].unique())
        print(f"Weeks available: {weeks}")
        print()

        # Count by week
        week_counts = stats_df.groupby('week').size()
        print("Player-games by week:")
        for week, count in week_counts.items():
            print(f"  Week {week:2d}: {count:4d} player-games")
        print()

    # Check position groups
    if 'position' in stats_df.columns:
        position_counts = stats_df['position'].value_counts()
        print("Players by position:")
        for pos, count in position_counts.head(10).items():
            print(f"  {pos}: {count}")
        print()

    # Check stat columns
    stat_columns = [
        'passing_yards', 'passing_tds', 'completions', 'attempts',
        'rushing_yards', 'rushing_tds', 'carries',
        'receiving_yards', 'receiving_tds', 'receptions', 'targets'
    ]

    print("Available stat columns:")
    for col in stat_columns:
        if col in stats_df.columns:
            non_null = stats_df[col].notna().sum()
            pct = non_null / len(stats_df) * 100
            print(f"  ✅ {col:20s}: {non_null:6d} ({pct:5.1f}%)")
        else:
            print(f"  ❌ {col:20s}: Not available")

    print()

    return stats_df


def create_prop_training_data(stats_df, training_weeks):
    """Create training datasets for each prop type.

    For each player-game, we need:
    - Historical stats (up to that week)
    - Game context
    - Actual outcome
    """

    print(f"\n{'='*80}")
    print(f"CREATING PROP TRAINING DATA")
    print(f"{'='*80}\n")

    print(f"Training weeks: {training_weeks}")
    print()

    # Filter to training weeks
    train_df = stats_df[stats_df['week'].isin(training_weeks)].copy()

    print(f"Training records: {len(train_df)}")
    print()

    # For each prop type, create features
    prop_datasets = {}

    # 1. Pass Yards
    if 'passing_yards' in train_df.columns:
        pass_df = train_df[train_df['passing_yards'].notna()].copy()

        print(f"📊 Pass Yards props: {len(pass_df)} QB performances")

        # Calculate rolling averages for each QB
        pass_df = pass_df.sort_values(['player_id', 'week'])

        # Season average (up to this week)
        pass_df['season_avg_pass_yards'] = pass_df.groupby('player_id')['passing_yards'].expanding().mean().reset_index(level=0, drop=True)

        # Last 3 games average
        pass_df['l3_avg_pass_yards'] = pass_df.groupby('player_id')['passing_yards'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

        # Games played
        pass_df['games_played'] = pass_df.groupby('player_id').cumcount() + 1

        prop_datasets['pass_yards'] = pass_df

        print(f"   Features: season_avg, l3_avg, games_played")
        print()

    # 2. Rush Yards
    if 'rushing_yards' in train_df.columns:
        rush_df = train_df[train_df['rushing_yards'].notna()].copy()
        rush_df = rush_df[rush_df['carries'] > 0].copy()  # Only RBs with carries

        print(f"📊 Rush Yards props: {len(rush_df)} RB performances")

        rush_df = rush_df.sort_values(['player_id', 'week'])

        rush_df['season_avg_rush_yards'] = rush_df.groupby('player_id')['rushing_yards'].expanding().mean().reset_index(level=0, drop=True)
        rush_df['l3_avg_rush_yards'] = rush_df.groupby('player_id')['rushing_yards'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        rush_df['games_played'] = rush_df.groupby('player_id').cumcount() + 1

        prop_datasets['rush_yards'] = rush_df

        print(f"   Features: season_avg, l3_avg, games_played")
        print()

    # 3. Rec Yards
    if 'receiving_yards' in train_df.columns:
        rec_df = train_df[train_df['receiving_yards'].notna()].copy()
        rec_df = rec_df[rec_df['targets'] > 0].copy()  # Only pass catchers

        print(f"📊 Rec Yards props: {len(rec_df)} WR/TE/RB performances")

        rec_df = rec_df.sort_values(['player_id', 'week'])

        rec_df['season_avg_rec_yards'] = rec_df.groupby('player_id')['receiving_yards'].expanding().mean().reset_index(level=0, drop=True)
        rec_df['l3_avg_rec_yards'] = rec_df.groupby('player_id')['receiving_yards'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        rec_df['games_played'] = rec_df.groupby('player_id').cumcount() + 1

        prop_datasets['rec_yards'] = rec_df

        print(f"   Features: season_avg, l3_avg, games_played")
        print()

    return prop_datasets


def save_training_datasets(prop_datasets):
    """Save training data for each prop type."""

    outputs_dir = Path('outputs/training')
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving training datasets:")
    print()

    for prop_type, df in prop_datasets.items():
        output_file = outputs_dir / f'{prop_type}_training.csv'

        df.to_csv(output_file, index=False)

        print(f"  ✅ {prop_type}: {len(df)} records → {output_file}")

    print()


def main():
    """Main pipeline."""

    # Try to use synthetic data first
    synthetic_file = Path('inputs/player_stats_2025_synthetic.csv')

    if synthetic_file.exists():
        print(f"✅ Using synthetic player stats from {synthetic_file}\n")
        stats_df = pd.read_csv(synthetic_file)
        year = 2025
        training_weeks = list(range(1, 10))

    else:
        # Download 2025 player stats
        stats_2025 = download_player_stats(2025)

        if len(stats_2025) == 0:
            print("❌ No player stats available for 2025")
            print("   Trying 2024 instead...")

            stats_2024 = download_player_stats(2024)

            if len(stats_2024) == 0:
                print("❌ No player stats available")
                print("   Run: python -m backend.analysis.generate_synthetic_player_stats")
                return

            stats_df = stats_2024
            year = 2024
            training_weeks = list(range(1, 10))

        else:
            stats_df = stats_2025
            year = 2025
            training_weeks = list(range(1, 10))

    # Analyze stats
    stats_df = analyze_player_stats(stats_df)

    # Create training data
    prop_datasets = create_prop_training_data(stats_df, training_weeks)

    # Save datasets
    save_training_datasets(prop_datasets)

    print(f"\n{'='*80}")
    print(f"✅ PLAYER STATS EXTRACTION COMPLETE")
    print(f"{'='*80}\n")

    print(f"Next steps:")
    print(f"  1. Train models on outputs/training/{'{prop_type}'}_training.csv")
    print(f"  2. Make predictions for Week 10")
    print(f"  3. Compare to actual results")
    print()


if __name__ == '__main__':
    main()
