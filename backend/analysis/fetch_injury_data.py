"""Fetch and process injury/active status data from NFLverse.

Critical for prop betting - cannot bet on inactive players!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from typing import Dict, Set


def fetch_weekly_rosters(year: int) -> pd.DataFrame:
    """Fetch weekly roster data from NFLverse.

    Args:
        year: NFL season year

    Returns:
        DataFrame with columns: player_id, player_name, team, week, status, position
    """

    print(f"\n{'='*80}")
    print(f"FETCHING WEEKLY ROSTERS - {year}")
    print(f"{'='*80}\n")

    # NFLverse weekly roster URL
    url = f"https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{year}.csv"

    print(f"Downloading from: {url}")

    try:
        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            import io
            rosters_df = pd.read_csv(io.StringIO(response.text))

            print(f"✅ Downloaded {len(rosters_df)} player-week records")
            print(f"   Weeks available: {sorted(rosters_df['week'].unique())}")
            print()

            # Show status breakdown
            if 'status' in rosters_df.columns:
                status_counts = rosters_df['status'].value_counts()
                print("Status breakdown:")
                for status, count in status_counts.items():
                    print(f"  {status}: {count}")
                print()

            return rosters_df

        else:
            print(f"❌ Failed to download: HTTP {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")

    return pd.DataFrame()


def get_inactive_players(rosters_df: pd.DataFrame, week: int) -> Set[str]:
    """Get set of inactive player IDs for a specific week.

    Args:
        rosters_df: Weekly rosters DataFrame
        week: Week number

    Returns:
        Set of player_ids who are OUT/DOUBTFUL/IR/PUP/etc.
    """

    # Filter to week
    week_rosters = rosters_df[rosters_df['week'] == week].copy()

    # Define inactive statuses
    inactive_statuses = {'OUT', 'IR', 'PUP', 'DNR', 'NON', 'SUS'}

    # Get inactive players
    inactive = week_rosters[week_rosters['status'].isin(inactive_statuses)]['gsis_id'].unique()

    return set(inactive)


def get_questionable_players(rosters_df: pd.DataFrame, week: int) -> Set[str]:
    """Get set of questionable player IDs for a specific week.

    These players should have reduced confidence.

    Args:
        rosters_df: Weekly rosters DataFrame
        week: Week number

    Returns:
        Set of player_ids who are QUESTIONABLE
    """

    # Filter to week
    week_rosters = rosters_df[rosters_df['week'] == week].copy()

    # Questionable status
    questionable = week_rosters[week_rosters['status'] == 'Q']['gsis_id'].unique()

    return set(questionable)


def get_player_status_summary(rosters_df: pd.DataFrame, week: int) -> pd.DataFrame:
    """Get complete player status summary for a week.

    Args:
        rosters_df: Weekly rosters DataFrame
        week: Week number

    Returns:
        DataFrame with columns: player_id, player_name, team, position, status
    """

    week_rosters = rosters_df[rosters_df['week'] == week].copy()

    summary = week_rosters[[
        'gsis_id', 'full_name', 'team', 'position', 'status'
    ]].copy()

    summary.columns = ['player_id', 'player_name', 'team', 'position', 'status']

    return summary


def filter_predictions_by_status(
    predictions_df: pd.DataFrame,
    rosters_df: pd.DataFrame,
    week: int,
    remove_inactive: bool = True,
    penalize_questionable: bool = True
) -> pd.DataFrame:
    """Filter prop predictions based on player injury status.

    Args:
        predictions_df: DataFrame with prop predictions (must have 'player_id' column)
        rosters_df: Weekly rosters DataFrame
        week: Week number
        remove_inactive: If True, remove OUT/IR/etc players
        penalize_questionable: If True, reduce edge for QUESTIONABLE players

    Returns:
        Filtered DataFrame with status column added
    """

    print(f"\n{'='*80}")
    print(f"FILTERING PREDICTIONS BY INJURY STATUS - WEEK {week}")
    print(f"{'='*80}\n")

    print(f"Input predictions: {len(predictions_df)}")

    # Get player statuses
    inactive_ids = get_inactive_players(rosters_df, week)
    questionable_ids = get_questionable_players(rosters_df, week)

    print(f"Inactive players this week: {len(inactive_ids)}")
    print(f"Questionable players this week: {len(questionable_ids)}")
    print()

    # Add status column
    def get_status(player_id):
        if player_id in inactive_ids:
            return 'INACTIVE'
        elif player_id in questionable_ids:
            return 'QUESTIONABLE'
        else:
            return 'ACTIVE'

    predictions_df['injury_status'] = predictions_df['player_id'].apply(get_status)

    # Log removals
    if remove_inactive:
        removed = predictions_df[predictions_df['injury_status'] == 'INACTIVE']

        if len(removed) > 0:
            print(f"🚫 REMOVING {len(removed)} bets on INACTIVE players:")
            for idx, row in removed.iterrows():
                print(f"   ❌ {row['player_name']} ({row.get('prop_type', 'unknown')})")
            print()

        # Remove inactive
        predictions_df = predictions_df[predictions_df['injury_status'] != 'INACTIVE'].copy()

    # Penalize questionable
    if penalize_questionable and 'edge' in predictions_df.columns:
        questionable = predictions_df[predictions_df['injury_status'] == 'QUESTIONABLE']

        if len(questionable) > 0:
            print(f"⚠️  REDUCING edge by 50% for {len(questionable)} QUESTIONABLE players:")
            for idx, row in questionable.iterrows():
                print(f"   {row['player_name']} ({row.get('prop_type', 'unknown')})")
            print()

            # Reduce edge by 50%
            predictions_df.loc[
                predictions_df['injury_status'] == 'QUESTIONABLE',
                'edge'
            ] *= 0.5

    print(f"✅ Output predictions: {len(predictions_df)}")
    print(f"   Active: {(predictions_df['injury_status'] == 'ACTIVE').sum()}")
    print(f"   Questionable: {(predictions_df['injury_status'] == 'QUESTIONABLE').sum()}")
    print()

    return predictions_df


def main():
    """Demo: Fetch injury data for 2025 season."""

    # Fetch 2025 rosters
    rosters_2025 = fetch_weekly_rosters(2025)

    if len(rosters_2025) == 0:
        print("❌ No roster data available for 2025")
        print("   Trying 2024 instead...")

        rosters_2024 = fetch_weekly_rosters(2024)

        if len(rosters_2024) == 0:
            print("❌ No roster data available")
            return

        rosters_df = rosters_2024
        year = 2024
    else:
        rosters_df = rosters_2025
        year = 2025

    # Save rosters
    output_file = Path(f'inputs/weekly_rosters_{year}.csv')
    rosters_df.to_csv(output_file, index=False)

    print(f"💾 Saved rosters to: {output_file}")
    print()

    # Show Week 10 status summary
    week = 10

    print(f"\n{'='*80}")
    print(f"WEEK {week} STATUS SUMMARY")
    print(f"{'='*80}\n")

    inactive = get_inactive_players(rosters_df, week)
    questionable = get_questionable_players(rosters_df, week)

    print(f"INACTIVE players (OUT/IR/etc): {len(inactive)}")
    print(f"QUESTIONABLE players: {len(questionable)}")
    print()

    # Show some examples
    summary = get_player_status_summary(rosters_df, week)

    print("Sample INACTIVE players:")
    inactive_players = summary[summary['status'].isin(['OUT', 'IR', 'PUP'])].head(10)
    for idx, row in inactive_players.iterrows():
        print(f"  ❌ {row['player_name']:30s} ({row['position']:3s}, {row['team']}) - {row['status']}")
    print()

    print("Sample QUESTIONABLE players:")
    questionable_players = summary[summary['status'] == 'Q'].head(10)
    for idx, row in questionable_players.iterrows():
        print(f"  ⚠️  {row['player_name']:30s} ({row['position']:3s}, {row['team']}) - {row['status']}")
    print()


if __name__ == '__main__':
    main()
