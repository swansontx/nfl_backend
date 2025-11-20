"""Real prop backtest for November 9, 2025 games.

Loads real player stats from nflverse data and compares
model projections to actual results.

IMPORTANT: This file requires:
1. Real player stats in inputs/player_stats_2024_2025.csv
2. Historical prop lines from real sportsbook data

If you don't have real historical lines, you cannot run backtests.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Optional


def load_player_stats(inputs_dir: Path = Path("inputs")) -> pd.DataFrame:
    """Load real player stats from nflverse data."""
    stats_file = inputs_dir / "player_stats_2024_2025.csv"

    if not stats_file.exists():
        raise FileNotFoundError(
            f"Player stats file not found: {stats_file}\n"
            "Run: python -m backend.ingestion.fetch_nflverse --year 2024"
        )

    df = pd.read_csv(stats_file, low_memory=False)
    print(f"Loaded {len(df)} player stat records")
    return df


def load_historical_props(outputs_dir: Path = Path("outputs")) -> List[Dict]:
    """Load historical prop lines from real data.

    Returns list of prop dicts with player, prop_type, and line fields.
    These must be REAL sportsbook lines, not generated/simulated.
    """
    # Check for saved historical props file
    props_file = outputs_dir / "historical_props_week10_2024.json"

    if not props_file.exists():
        print(f"ERROR: Historical props file not found: {props_file}")
        print("You need real sportsbook lines to run backtests.")
        print("Fetch them using: python -m backend.ingestion.fetch_odds")
        return []

    with open(props_file) as f:
        data = json.load(f)

    return data.get('props', [])


def generate_projection(player_df: pd.DataFrame, prop_type: str) -> Optional[float]:
    """Generate projection for a player using their historical data.

    Uses weighted rolling average:
    - 50% last 3 games
    - 30% season average
    - 20% last game
    """
    if len(player_df) < 2:
        return None

    # Map prop types to stat columns
    stat_map = {
        'pass_yards': 'passing_yards',
        'passing_yards': 'passing_yards',
        'rush_yards': 'rushing_yards',
        'rushing_yards': 'rushing_yards',
        'rec_yards': 'receiving_yards',
        'receiving_yards': 'receiving_yards',
        'receptions': 'receptions',
        'completions': 'completions',
        'passing_tds': 'passing_tds',
        'rushing_tds': 'rushing_tds',
        'receiving_tds': 'receiving_tds',
    }

    col = stat_map.get(prop_type)
    if not col or col not in player_df.columns:
        return None

    values = player_df[col].dropna()
    if len(values) == 0:
        return None

    season_avg = values.mean()
    l3_avg = values.tail(3).mean() if len(values) >= 3 else season_avg
    last_game = values.iloc[-1] if len(values) > 0 else season_avg

    projection = 0.5 * l3_avg + 0.3 * season_avg + 0.2 * last_game

    return round(projection, 1)


def run_prop_backtest(week: int = 10, season: int = 2024):
    """Run complete prop backtest for a specific week.

    Requires:
    1. Real player stats from nflverse
    2. Real historical prop lines from sportsbooks
    """
    print(f"\n{'#'*80}")
    print(f"# PROP BACKTEST: Week {week}, {season}")
    print(f"{'#'*80}\n")

    inputs_dir = Path("inputs")
    outputs_dir = Path("outputs")

    # Load real player stats
    try:
        player_stats = load_player_stats(inputs_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Load real historical props (lines from sportsbooks)
    historical_props = load_historical_props(outputs_dir)

    if not historical_props:
        print("\nERROR: No historical prop lines available.")
        print("Backtests require REAL sportsbook lines, not simulated data.")
        print("\nTo get historical lines:")
        print("1. Fetch current odds: python -m backend.ingestion.fetch_odds")
        print("2. Save the lines before games start")
        print("3. Store them in outputs/historical_props_week{N}_{year}.json")
        return

    print(f"Loaded {len(historical_props)} historical prop lines")

    # Get data BEFORE target week (for projections)
    historical_data = player_stats[player_stats['week'] < week].copy()

    # Get actual results FROM target week
    actual_data = player_stats[player_stats['week'] == week].copy()

    print(f"Historical data (weeks 1-{week-1}): {len(historical_data)} records")
    print(f"Actual results (week {week}): {len(actual_data)} records")

    if len(actual_data) == 0:
        print(f"ERROR: No actual data found for week {week}")
        return

    # Generate projections and compare to actuals
    results = []

    for prop in historical_props:
        player_name = prop['player']
        prop_type = prop['prop_type']
        line = prop['line']

        # Find player in historical data
        player_historical = historical_data[
            historical_data['player_display_name'] == player_name
        ].sort_values('week')

        if len(player_historical) < 2:
            continue

        # Generate projection using historical data
        projection = generate_projection(player_historical, prop_type)
        if projection is None:
            continue

        # Get actual result
        player_actual = actual_data[
            actual_data['player_display_name'] == player_name
        ]

        if len(player_actual) == 0:
            continue

        # Map prop type to actual stat column
        stat_map = {
            'pass_yards': 'passing_yards',
            'passing_yards': 'passing_yards',
            'rush_yards': 'rushing_yards',
            'rushing_yards': 'rushing_yards',
            'rec_yards': 'receiving_yards',
            'receiving_yards': 'receiving_yards',
            'receptions': 'receptions',
        }

        stat_col = stat_map.get(prop_type)
        if not stat_col:
            continue

        actual = player_actual.iloc[0].get(stat_col, 0)
        if pd.isna(actual):
            continue

        # Determine if bet would hit
        side = "OVER" if projection > line else "UNDER"
        hit = (side == "OVER" and actual > line) or (side == "UNDER" and actual < line)

        results.append({
            'player': player_name,
            'team': player_historical.iloc[-1].get('team', 'UNK'),
            'prop_type': prop_type,
            'line': line,
            'side': side,
            'projection': projection,
            'actual': actual,
            'hit': hit,
            'edge': round((projection - line) / line * 100, 1),
        })

        status = "HIT" if hit else "MISS"
        print(f"  {status} - {player_name} {prop_type} {side} {line}")
        print(f"    Predicted: {projection:.1f} | Actual: {actual:.0f}")

    if not results:
        print("\nNo results to analyze")
        return

    # Calculate performance
    results_df = pd.DataFrame(results)
    total = len(results_df)
    wins = results_df['hit'].sum()
    win_rate = wins / total if total > 0 else 0

    # ROI calculation (standard -110 odds)
    total_staked = total * 110
    winnings = wins * 100
    profit = winnings - (total - wins) * 110
    roi = profit / total_staked if total_staked > 0 else 0

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total props: {total}")
    print(f"Winners: {wins}")
    print(f"Losers: {total - wins}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"ROI: {roi:+.1%}")
    print(f"Profit: ${profit:+.0f}")

    # Save results
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_file = outputs_dir / f'backtest_props_week{week}_{season}.json'

    report = {
        'week': week,
        'season': season,
        'props': results,
        'metrics': {
            'total': total,
            'wins': int(wins),
            'losses': int(total - wins),
            'win_rate': float(win_rate),
            'roi': float(roi),
            'profit': float(profit),
        }
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {report_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Backtest props for a week")
    parser.add_argument("--week", type=int, default=10, help="Week to backtest")
    parser.add_argument("--season", type=int, default=2024, help="Season")

    args = parser.parse_args()

    run_prop_backtest(week=args.week, season=args.season)
