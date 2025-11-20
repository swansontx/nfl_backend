#!/usr/bin/env python3
"""Quick picks tool for comparing projections against DraftKings lines.

Usage:
    python -m backend.tools.quick_picks --week 12 --teams PIT,CLE
"""

import pandas as pd
import argparse
from pathlib import Path
from math import erf, sqrt


def load_projections(week: int, season: int = 2025) -> pd.DataFrame:
    """Load projections for a given week."""
    proj_file = Path(f"outputs/predictions/props_{season}_{week}.csv")
    if not proj_file.exists():
        print(f"Error: {proj_file} not found. Run generate_projections first.")
        return pd.DataFrame()
    return pd.read_csv(proj_file)


def filter_by_teams(df: pd.DataFrame, teams: list) -> pd.DataFrame:
    """Filter projections by team."""
    teams_upper = [t.upper() for t in teams]
    return df[df['team'].isin(teams_upper)]


def calculate_edge(projection: float, std_dev: float, line: float, odds: int = -110) -> dict:
    """Calculate edge for over/under.

    Args:
        projection: Our projected value
        std_dev: Standard deviation
        line: DraftKings line
        odds: American odds (default -110)

    Returns:
        Dict with prob_over, prob_under, edge_over, edge_under, recommendation
    """
    # Calculate probability using normal CDF
    if std_dev > 0:
        z_score = (line - projection) / std_dev
        prob_under = 0.5 * (1 + erf(z_score / sqrt(2)))
        prob_over = 1 - prob_under
    else:
        prob_over = 1.0 if projection > line else 0.0
        prob_under = 1 - prob_over

    # Convert odds to implied probability
    if odds < 0:
        implied_prob = abs(odds) / (abs(odds) + 100)
    else:
        implied_prob = 100 / (odds + 100)

    # Calculate edge
    edge_over = (prob_over - implied_prob) * 100
    edge_under = (prob_under - implied_prob) * 100

    # Recommendation
    if edge_over > 5:
        rec = "OVER"
        grade = "A" if edge_over > 10 else "B"
    elif edge_under > 5:
        rec = "UNDER"
        grade = "A" if edge_under > 10 else "B"
    else:
        rec = "PASS"
        grade = "C"

    return {
        'prob_over': round(prob_over * 100, 1),
        'prob_under': round(prob_under * 100, 1),
        'edge_over': round(edge_over, 1),
        'edge_under': round(edge_under, 1),
        'recommendation': rec,
        'grade': grade
    }


def display_projections(df: pd.DataFrame, teams: list):
    """Display projections in a nice format."""
    print(f"\n{'='*70}")
    print(f"PROJECTIONS FOR: {', '.join(teams)}")
    print(f"{'='*70}\n")

    # Group by player
    for player_name in df['player_name'].unique():
        player_df = df[df['player_name'] == player_name]
        if len(player_df) == 0:
            continue

        team = player_df.iloc[0]['team']
        pos = player_df.iloc[0]['position']

        print(f"\n{player_name} ({team} {pos})")
        print("-" * 50)

        for _, row in player_df.iterrows():
            prop = row['prop_type']
            proj = row['projection']
            std = row['std_dev']
            low = row['confidence_lower']
            high = row['confidence_upper']

            print(f"  {prop:20s}: {proj:6.1f}  (range: {low:.0f}-{high:.0f})")


def interactive_mode(df: pd.DataFrame):
    """Interactive mode to compare against DK lines."""
    print("\n" + "="*70)
    print("INTERACTIVE LINE COMPARISON")
    print("="*70)
    print("Enter DraftKings lines to compare against projections.")
    print("Format: player_name prop_type line")
    print("Example: Najee Harris rushing_yards 65.5")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Enter line (or 'quit'): ").strip()
            if user_input.lower() == 'quit':
                break

            parts = user_input.rsplit(' ', 2)
            if len(parts) < 3:
                print("Invalid format. Use: player_name prop_type line")
                continue

            player_name = parts[0]
            prop_type = parts[1]
            line = float(parts[2])

            # Find matching projection
            match = df[
                (df['player_name'].str.lower().str.contains(player_name.lower())) &
                (df['prop_type'] == prop_type)
            ]

            if len(match) == 0:
                print(f"No projection found for {player_name} {prop_type}")
                continue

            row = match.iloc[0]
            projection = row['projection']
            std_dev = row['std_dev']

            # Calculate edge
            result = calculate_edge(projection, std_dev, line)

            print(f"\n  {row['player_name']} - {prop_type}")
            print(f"  Our Projection: {projection:.1f}")
            print(f"  DK Line: {line}")
            print(f"  Prob Over: {result['prob_over']}%  |  Prob Under: {result['prob_under']}%")
            print(f"  Edge Over: {result['edge_over']:+.1f}%  |  Edge Under: {result['edge_under']:+.1f}%")
            print(f"  >>> {result['recommendation']} ({result['grade']}) <<<\n")

        except ValueError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            break


def main():
    parser = argparse.ArgumentParser(description="Quick picks tool")
    parser.add_argument("--week", type=int, required=True, help="NFL week")
    parser.add_argument("--teams", type=str, help="Comma-separated team abbreviations (e.g., PIT,CLE)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Load projections
    df = load_projections(args.week)
    if len(df) == 0:
        return

    # Filter by teams if specified
    if args.teams:
        teams = [t.strip() for t in args.teams.split(',')]
        df = filter_by_teams(df, teams)

        if len(df) == 0:
            print(f"No projections found for teams: {args.teams}")
            return

        display_projections(df, teams)

    # Interactive mode
    if args.interactive:
        interactive_mode(df)
    elif not args.teams:
        print("Specify --teams or use --interactive mode")
        print("Example: python -m backend.tools.quick_picks --week 12 --teams PIT,CLE -i")


if __name__ == "__main__":
    main()
