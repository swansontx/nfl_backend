"""
Example: Comprehensive Team Analysis

Shows how to use the Unified Metrics API to analyze a team's performance
across all available metrics.

Usage:
    python examples/analyze_team.py KC
    python examples/analyze_team.py BUF --weeks 9 10 11 12
"""

import sys
import os
import argparse

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.metrics.unified_metrics_api import get_metrics_api


def analyze_team(team: str, weeks=None):
    """
    Generate comprehensive team analysis report.

    Args:
        team: Team abbreviation (e.g., 'KC', 'BUF')
        weeks: Optional list of weeks for recency
    """
    print("\n" + "="*70)
    print(f"TEAM ANALYSIS: {team}")
    if weeks:
        print(f"Weeks: {', '.join(map(str, weeks))}")
    else:
        print("Full Season")
    print("="*70)

    # Initialize API
    api = get_metrics_api(season=2025)

    # Get metrics
    metrics = api.get_team_metrics(team, weeks=weeks)

    if not metrics:
        print(f"\n❌ No data found for team '{team}'")
        return

    # Offensive Performance
    print("\n📈 OFFENSIVE PERFORMANCE")
    print("-" * 70)

    if 'points_per_game' in metrics:
        print(f"Points Per Game:        {metrics['points_per_game']:.1f}")

    if 'yards_per_game' in metrics:
        print(f"Yards Per Game:         {metrics['yards_per_game']:.1f}")

    if 'passing_yards_per_game' in metrics:
        print(f"  Passing YPG:          {metrics['passing_yards_per_game']:.1f}")

    if 'rushing_yards_per_game' in metrics:
        print(f"  Rushing YPG:          {metrics['rushing_yards_per_game']:.1f}")

    # Efficiency Metrics
    print("\n⚡ EFFICIENCY METRICS")
    print("-" * 70)

    if 'success_rate_offense' in metrics:
        print(f"Success Rate:           {metrics['success_rate_offense']:.1%}")

    if 'epa_per_play_offense' in metrics:
        print(f"EPA Per Play:           {metrics['epa_per_play_offense']:+.3f}")

    if 'completion_pct' in metrics:
        print(f"Completion %:           {metrics['completion_pct']:.1%}")

    if 'yards_per_attempt' in metrics:
        print(f"Yards Per Attempt:      {metrics['yards_per_attempt']:.2f}")

    if 'yards_per_carry' in metrics:
        print(f"Yards Per Carry:        {metrics['yards_per_carry']:.2f}")

    # Pace
    print("\n⏱️  PACE & TIME OF POSSESSION")
    print("-" * 70)

    if 'plays_per_game' in metrics:
        pace = metrics['plays_per_game']
        print(f"Plays Per Game:         {pace:.1f}")

        # Compare to league average
        if pace > 68:
            print(f"                        🔥 FAST pace (league avg ~65)")
        elif pace < 62:
            print(f"                        🐢 SLOW pace (league avg ~65)")
        else:
            print(f"                        ➖ AVERAGE pace")

    if 'time_of_possession_pct' in metrics:
        top = metrics['time_of_possession_pct']
        print(f"Time of Possession:     {top:.1%}")

    # Turnovers
    print("\n💥 TURNOVER METRICS")
    print("-" * 70)

    if 'turnover_margin' in metrics:
        margin = metrics['turnover_margin']
        print(f"Turnover Margin:        {margin:+d}")

        if margin >= 5:
            print(f"                        ✅ EXCELLENT ball security")
        elif margin >= 0:
            print(f"                        ✅ Positive margin")
        elif margin >= -5:
            print(f"                        ⚠️  Negative margin")
        else:
            print(f"                        ❌ POOR ball security")

    if 'turnover_rate' in metrics:
        print(f"Turnover Rate:          {metrics['turnover_rate']:.2%} (per 100 plays)")

    if 'takeaway_rate' in metrics:
        print(f"Takeaway Rate:          {metrics['takeaway_rate']:.2%} (per 100 plays)")

    # Red Zone
    print("\n🎯 RED ZONE PERFORMANCE")
    print("-" * 70)

    if 'red_zone_td_pct' in metrics:
        rz_td = metrics['red_zone_td_pct']
        print(f"TD %:                   {rz_td:.1%}")

        if rz_td >= 0.60:
            print(f"                        🔥 ELITE red zone offense")
        elif rz_td >= 0.50:
            print(f"                        ✅ GOOD red zone offense")
        else:
            print(f"                        ⚠️  Below average")

    if 'red_zone_score_pct' in metrics:
        print(f"Score %:                {metrics['red_zone_score_pct']:.1%}")

    if 'red_zone_attempts' in metrics:
        print(f"Attempts:               {metrics['red_zone_attempts']:.0f}")

    # Third Down
    print("\n🏈 THIRD DOWN EFFICIENCY")
    print("-" * 70)

    if 'third_down_pct' in metrics:
        third_down = metrics['third_down_pct']
        print(f"Conversion Rate:        {third_down:.1%}")

        if third_down >= 0.45:
            print(f"                        ✅ EXCELLENT (top tier)")
        elif third_down >= 0.40:
            print(f"                        ✅ GOOD")
        elif third_down >= 0.35:
            print(f"                        ➖ AVERAGE")
        else:
            print(f"                        ⚠️  POOR")

    if 'third_down_attempts' in metrics:
        print(f"Attempts:               {metrics['third_down_attempts']:.0f}")

    # Explosive Plays
    print("\n💨 EXPLOSIVE PLAYS")
    print("-" * 70)

    if 'explosive_play_rate' in metrics:
        explosive = metrics['explosive_play_rate']
        print(f"Explosive Play Rate:    {explosive:.1%}")
        print(f"                        (20+ yd pass, 10+ yd rush)")

        if explosive >= 0.12:
            print(f"                        🔥 HIGH big play rate")
        elif explosive >= 0.08:
            print(f"                        ✅ AVERAGE")
        else:
            print(f"                        ⚠️  LOW big play rate")

    # Defensive Performance
    print("\n🛡️  DEFENSIVE PERFORMANCE")
    print("-" * 70)

    if 'points_allowed_per_game' in metrics:
        ppg_allowed = metrics.get('points_allowed_per_game', 0)
        print(f"Points Allowed:         {ppg_allowed:.1f}")

        if ppg_allowed <= 18:
            print(f"                        🔥 ELITE defense")
        elif ppg_allowed <= 22:
            print(f"                        ✅ GOOD defense")
        elif ppg_allowed <= 26:
            print(f"                        ➖ AVERAGE")
        else:
            print(f"                        ⚠️  POOR defense")

    if 'success_rate_defense' in metrics:
        print(f"Success Rate Allowed:   {metrics['success_rate_defense']:.1%}")

    if 'epa_per_play_defense' in metrics:
        print(f"EPA Allowed:            {metrics['epa_per_play_defense']:+.3f}")

    if 'explosive_plays_allowed_rate' in metrics:
        print(f"Explosive Plays Allowed: {metrics['explosive_plays_allowed_rate']:.1%}")

    # Home/Away Splits
    if 'home_ppg' in metrics and 'away_ppg' in metrics:
        print("\n🏟️  HOME/AWAY SPLITS")
        print("-" * 70)
        print(f"Home PPG:               {metrics['home_ppg']:.1f}")
        print(f"Away PPG:               {metrics['away_ppg']:.1f}")
        print(f"Differential:           {metrics['home_ppg'] - metrics['away_ppg']:+.1f}")

    # Summary
    print("\n" + "="*70)
    print(f"Total Metrics Available: {len(metrics)}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze NFL team metrics")
    parser.add_argument("team", help="Team abbreviation (e.g., KC, BUF)")
    parser.add_argument("--weeks", nargs="+", type=int,
                        help="Specific weeks to analyze (e.g., --weeks 9 10 11 12)")

    args = parser.parse_args()

    try:
        analyze_team(args.team, weeks=args.weeks)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
