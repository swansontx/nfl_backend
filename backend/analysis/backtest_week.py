"""Backtest projections against actual results for a completed week.

This script:
1. Generates projections for a week using data available BEFORE that week
2. Loads actual results from that week
3. Compares projections to actuals
4. Calculates hit rates and accuracy metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict


def load_player_stats(inputs_dir: Path) -> pd.DataFrame:
    """Load player stats."""
    enhanced_file = inputs_dir / "player_stats_enhanced_2025.csv"
    if enhanced_file.exists():
        return pd.read_csv(enhanced_file, low_memory=False)

    basic_file = inputs_dir / "player_stats_2025.csv"
    if basic_file.exists():
        return pd.read_csv(basic_file, low_memory=False)

    raise FileNotFoundError("No player stats file found")


def get_player_projection(
    player_df: pd.DataFrame,
    stat_type: str
) -> Tuple[float, float]:
    """Get projection for a player using data before target week.

    Uses simple rolling average as baseline projection.

    Returns:
        (projection, std_dev)
    """
    if len(player_df) == 0:
        return 0.0, 0.0

    # Map stat types to column names
    stat_map = {
        'passing_yards': 'passing_yards',
        'rushing_yards': 'rushing_yards',
        'receiving_yards': 'receiving_yards',
        'receptions': 'receptions',
        'completions': 'completions',
        'attempts': 'attempts',
        'carries': 'carries',
        'targets': 'targets',
        'passing_tds': 'passing_tds',
        'rushing_tds': 'rushing_tds',
        'receiving_tds': 'receiving_tds',
    }

    col = stat_map.get(stat_type)
    if not col or col not in player_df.columns:
        return 0.0, 0.0

    values = player_df[col].dropna()
    if len(values) == 0:
        return 0.0, 0.0

    # Use last 3 games average as projection
    recent_values = values.tail(3)
    projection = recent_values.mean()

    # Std dev from all games
    std_dev = values.std() if len(values) > 1 else projection * 0.2

    return projection, std_dev


def backtest_week(week: int, season: int = 2025) -> Dict:
    """Run backtest for a specific week.

    Args:
        week: Week to backtest
        season: Season year

    Returns:
        Dict with backtest results
    """
    inputs_dir = Path("inputs")

    print(f"\n{'='*60}")
    print(f"BACKTESTING WEEK {week}")
    print(f"{'='*60}\n")

    # Load player stats
    player_stats = load_player_stats(inputs_dir)

    # Get data BEFORE target week (for predictions)
    historical_data = player_stats[player_stats['week'] < week].copy()

    # Get actual results FROM target week
    actual_data = player_stats[player_stats['week'] == week].copy()

    print(f"Historical data (weeks 1-{week-1}): {len(historical_data)} records")
    print(f"Actual results (week {week}): {len(actual_data)} records")

    if len(actual_data) == 0:
        print(f"No data for week {week}")
        return {}

    # Define prop types to backtest
    prop_types = {
        'QB': ['passing_yards', 'completions', 'attempts', 'passing_tds'],
        'RB': ['rushing_yards', 'carries', 'receptions', 'receiving_yards', 'rushing_tds'],
        'WR': ['receptions', 'receiving_yards', 'targets', 'receiving_tds'],
        'TE': ['receptions', 'receiving_yards', 'targets', 'receiving_tds'],
    }

    # Generate projections and compare to actuals
    results = []

    for _, actual_row in actual_data.iterrows():
        player_id = actual_row['player_id']
        player_name = actual_row.get('player_display_name', player_id)
        position = actual_row.get('position', 'UNK')

        # Get player's historical data
        player_historical = historical_data[
            historical_data['player_id'] == player_id
        ].sort_values('week')

        if len(player_historical) < 2:
            continue

        # Get position group
        pos_group = 'QB' if 'QB' in str(position) else \
                   'RB' if 'RB' in str(position) else \
                   'WR' if 'WR' in str(position) else \
                   'TE' if 'TE' in str(position) else None

        if not pos_group:
            continue

        # Backtest each prop type for this position
        for prop_type in prop_types.get(pos_group, []):
            projection, std_dev = get_player_projection(player_historical, prop_type)

            if projection <= 0:
                continue

            # Get actual value
            stat_col = {
                'passing_yards': 'passing_yards',
                'rushing_yards': 'rushing_yards',
                'receiving_yards': 'receiving_yards',
                'receptions': 'receptions',
                'completions': 'completions',
                'attempts': 'attempts',
                'carries': 'carries',
                'targets': 'targets',
                'passing_tds': 'passing_tds',
                'rushing_tds': 'rushing_tds',
                'receiving_tds': 'receiving_tds',
            }.get(prop_type)

            if not stat_col or stat_col not in actual_row:
                continue

            actual = actual_row[stat_col]
            if pd.isna(actual):
                continue

            # Calculate accuracy metrics
            error = actual - projection
            pct_error = (error / projection * 100) if projection > 0 else 0

            # Load real historical line if available
            # For now, skip props without real lines
            # TODO: Load from outputs/backtest_props_*.json when available
            historical_props_file = Path(f"outputs/backtest_props_nov9_2025.json")
            line = None

            if historical_props_file.exists():
                try:
                    import json
                    with open(historical_props_file) as f:
                        props_data = json.load(f)
                    for prop in props_data.get('props', []):
                        if prop['player'] == player_name and prop['prop_type'] == prop_type:
                            line = prop['line']
                            break
                except Exception:
                    pass

            if line is None:
                # No real line available - skip this prop
                continue

            # Would "over" have hit?
            over_hits = actual > line
            under_hits = actual < line

            # Edge calculation (simplified)
            # If projection > line, we'd bet over
            predicted_side = "OVER" if projection > line else "UNDER"
            hit = (predicted_side == "OVER" and over_hits) or \
                  (predicted_side == "UNDER" and under_hits)

            results.append({
                'player_id': player_id,
                'player_name': player_name,
                'position': position,
                'prop_type': prop_type,
                'projection': round(projection, 1),
                'std_dev': round(std_dev, 1),
                'line': line,
                'actual': actual,
                'error': round(error, 1),
                'pct_error': round(pct_error, 1),
                'predicted_side': predicted_side,
                'hit': hit,
            })

    # Analyze results
    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No results to analyze")
        return {}

    print(f"\nTotal predictions: {len(df)}")

    # Overall hit rate
    hit_rate = df['hit'].mean()
    print(f"Overall hit rate: {hit_rate*100:.1f}%")

    # By prop type
    print(f"\nHit rate by prop type:")
    by_prop = df.groupby('prop_type').agg({
        'hit': ['sum', 'count', 'mean'],
        'error': ['mean', 'std'],
    }).round(2)

    for prop_type in by_prop.index:
        hits = by_prop.loc[prop_type, ('hit', 'sum')]
        total = by_prop.loc[prop_type, ('hit', 'count')]
        rate = by_prop.loc[prop_type, ('hit', 'mean')]
        mae = abs(df[df['prop_type'] == prop_type]['error']).mean()
        print(f"  {prop_type:20s}: {rate*100:5.1f}% ({int(hits)}/{int(total)}) | MAE: {mae:.1f}")

    # By position
    print(f"\nHit rate by position:")
    by_pos = df.groupby('position').agg({
        'hit': ['sum', 'count', 'mean'],
    }).round(2)

    for pos in by_pos.index:
        hits = by_pos.loc[pos, ('hit', 'sum')]
        total = by_pos.loc[pos, ('hit', 'count')]
        rate = by_pos.loc[pos, ('hit', 'mean')]
        print(f"  {pos:5s}: {rate*100:5.1f}% ({int(hits)}/{int(total)})")

    # Top hits and misses
    print(f"\nTop 10 best predictions (closest to actual):")
    df['abs_error'] = df['error'].abs()
    top_hits = df.nsmallest(10, 'abs_error')
    for _, row in top_hits.iterrows():
        print(f"  {row['player_name']:20s} {row['prop_type']:15s}: Proj={row['projection']:6.1f}, Actual={row['actual']:6.1f}, Error={row['error']:+6.1f}")

    print(f"\nTop 10 worst predictions (furthest from actual):")
    worst_misses = df.nlargest(10, 'abs_error')
    for _, row in worst_misses.iterrows():
        print(f"  {row['player_name']:20s} {row['prop_type']:15s}: Proj={row['projection']:6.1f}, Actual={row['actual']:6.1f}, Error={row['error']:+6.1f}")

    # Identify bugs/patterns
    print(f"\n{'='*60}")
    print("POTENTIAL ISSUES TO FIX")
    print(f"{'='*60}")

    # Props with <45% hit rate
    poor_props = by_prop[by_prop[('hit', 'mean')] < 0.45].index.tolist()
    if poor_props:
        print(f"\nProp types with <45% hit rate: {poor_props}")

    # Large systematic errors
    for prop_type in df['prop_type'].unique():
        prop_df = df[df['prop_type'] == prop_type]
        mean_error = prop_df['error'].mean()
        if abs(mean_error) > 10:
            direction = "OVER" if mean_error > 0 else "UNDER"
            print(f"\n{prop_type}: Systematic {direction}-prediction by {abs(mean_error):.1f}")

    # Return summary
    return {
        'week': week,
        'total_predictions': len(df),
        'overall_hit_rate': round(hit_rate, 3),
        'by_prop_type': by_prop.to_dict(),
        'by_position': by_pos.to_dict(),
        'results': df.to_dict('records'),
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest projections for a week")
    parser.add_argument("--week", type=int, required=True, help="Week to backtest")
    parser.add_argument("--season", type=int, default=2025, help="Season year")

    args = parser.parse_args()

    results = backtest_week(args.week, args.season)

    if results:
        print(f"\n\nBacktest complete for week {args.week}")
        print(f"Overall hit rate: {results['overall_hit_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
