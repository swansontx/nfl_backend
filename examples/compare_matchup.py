"""
Example: Head-to-Head Matchup Comparison

Shows how to compare two teams and analyze their matchup using
the Unified Metrics API.

Usage:
    python examples/compare_matchup.py KC BUF
    python examples/compare_matchup.py KC BUF --week 13
"""

import sys
import os
import argparse

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.metrics.unified_metrics_api import get_metrics_api


def compare_matchup(home_team: str, away_team: str, week: int = None):
    """
    Generate head-to-head matchup comparison.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        week: Optional week number for game metrics
    """
    print("\n" + "="*70)
    print(f"MATCHUP ANALYSIS: {home_team} vs {away_team}")
    if week:
        print(f"Week: {week}")
    print("="*70)

    # Initialize API
    api = get_metrics_api(season=2025)

    # Get team comparison
    comparison = api.compare_teams(home_team, away_team)

    # Offensive Comparison
    print("\n⚔️  OFFENSIVE COMPARISON")
    print("-" * 70)

    offensive_metrics = [
        ('points_per_game', 'Points Per Game'),
        ('yards_per_game', 'Yards Per Game'),
        ('success_rate_offense', 'Success Rate'),
        ('epa_per_play_offense', 'EPA/Play'),
        ('yards_per_attempt', 'Yards/Attempt'),
        ('yards_per_carry', 'Yards/Carry'),
    ]

    for metric_key, label in offensive_metrics:
        if metric_key in comparison['metrics']:
            data = comparison['metrics'][metric_key]
            home_val = data.get(home_team, 0)
            away_val = data.get(away_team, 0)
            diff = data.get('difference', 0)

            # Determine advantage
            if abs(diff) < 0.01:
                adv_symbol = "="
            elif diff > 0:
                adv_symbol = f"✓ {home_team}"
            else:
                adv_symbol = f"✓ {away_team}"

            # Format values
            if isinstance(home_val, float) and 0 < abs(home_val) < 1:
                print(f"{label:20} {home_val:>7.1%} vs {away_val:>7.1%}  {adv_symbol}")
            else:
                print(f"{label:20} {home_val:>7.2f} vs {away_val:>7.2f}  {adv_symbol}")

    # Defensive Comparison
    print("\n🛡️  DEFENSIVE COMPARISON")
    print("-" * 70)

    defensive_metrics = [
        ('points_allowed_per_game', 'Points Allowed'),
        ('success_rate_defense', 'Success Rate Allowed'),
        ('epa_per_play_defense', 'EPA Allowed'),
        ('explosive_plays_allowed_rate', 'Explosive Plays Allowed'),
    ]

    for metric_key, label in defensive_metrics:
        if metric_key in comparison['metrics']:
            data = comparison['metrics'][metric_key]
            home_val = data.get(home_team, 0)
            away_val = data.get(away_team, 0)
            diff = data.get('difference', 0)

            # For defensive stats, lower is better (inverted)
            if abs(diff) < 0.01:
                adv_symbol = "="
            elif diff < 0:  # Home has lower (better)
                adv_symbol = f"✓ {home_team}"
            else:
                adv_symbol = f"✓ {away_team}"

            # Format values
            if isinstance(home_val, float) and 0 < abs(home_val) < 1:
                print(f"{label:25} {home_val:>7.1%} vs {away_val:>7.1%}  {adv_symbol}")
            else:
                print(f"{label:25} {home_val:>7.2f} vs {away_val:>7.2f}  {adv_symbol}")

    # Pace & Turnovers
    print("\n⏱️  PACE & TURNOVERS")
    print("-" * 70)

    pace_to_metrics = [
        ('plays_per_game', 'Plays Per Game'),
        ('turnover_margin', 'Turnover Margin'),
        ('turnover_rate', 'Turnover Rate'),
        ('takeaway_rate', 'Takeaway Rate'),
    ]

    for metric_key, label in pace_to_metrics:
        if metric_key in comparison['metrics']:
            data = comparison['metrics'][metric_key]
            home_val = data.get(home_team, 0)
            away_val = data.get(away_team, 0)
            diff = data.get('difference', 0)

            # Determine advantage
            if abs(diff) < 0.01:
                adv_symbol = "="
            elif diff > 0:
                adv_symbol = f"✓ {home_team}"
            else:
                adv_symbol = f"✓ {away_team}"

            # Format values
            if metric_key in ['turnover_rate', 'takeaway_rate']:
                print(f"{label:20} {home_val:>7.2%} vs {away_val:>7.2%}  {adv_symbol}")
            elif metric_key == 'turnover_margin':
                print(f"{label:20} {home_val:>+7.0f} vs {away_val:>+7.0f}  {adv_symbol}")
            else:
                print(f"{label:20} {home_val:>7.1f} vs {away_val:>7.1f}  {adv_symbol}")

    # Red Zone & Third Down
    print("\n🎯 SITUATIONAL EFFICIENCY")
    print("-" * 70)

    situational_metrics = [
        ('red_zone_td_pct', 'Red Zone TD %'),
        ('third_down_pct', 'Third Down %'),
        ('explosive_play_rate', 'Explosive Play Rate'),
    ]

    for metric_key, label in situational_metrics:
        if metric_key in comparison['metrics']:
            data = comparison['metrics'][metric_key]
            home_val = data.get(home_team, 0)
            away_val = data.get(away_team, 0)
            diff = data.get('difference', 0)

            # Determine advantage
            if abs(diff) < 0.01:
                adv_symbol = "="
            elif diff > 0:
                adv_symbol = f"✓ {home_team}"
            else:
                adv_symbol = f"✓ {away_team}"

            print(f"{label:25} {home_val:>7.1%} vs {away_val:>7.1%}  {adv_symbol}")

    # Advantages Summary
    print("\n📊 ADVANTAGES SUMMARY")
    print("-" * 70)

    home_advantages = comparison['advantages_a']
    away_advantages = comparison['advantages_b']

    print(f"\n{home_team} has {len(home_advantages)} significant advantages:")
    for metric in home_advantages[:8]:  # Show top 8
        print(f"  • {metric}")
    if len(home_advantages) > 8:
        print(f"  ... and {len(home_advantages) - 8} more")

    print(f"\n{away_team} has {len(away_advantages)} significant advantages:")
    for metric in away_advantages[:8]:  # Show top 8
        print(f"  • {metric}")
    if len(away_advantages) > 8:
        print(f"  ... and {len(away_advantages) - 8} more")

    # Overall Assessment
    print("\n🎲 MATCHUP ASSESSMENT")
    print("-" * 70)

    home_count = len(home_advantages)
    away_count = len(away_advantages)

    if home_count > away_count * 1.5:
        print(f"✅ {home_team} has a SIGNIFICANT statistical edge ({home_count} vs {away_count})")
    elif away_count > home_count * 1.5:
        print(f"✅ {away_team} has a SIGNIFICANT statistical edge ({away_count} vs {home_count})")
    elif home_count > away_count:
        print(f"✅ {home_team} has a SLIGHT edge ({home_count} vs {away_count})")
    elif away_count > home_count:
        print(f"✅ {away_team} has a SLIGHT edge ({away_count} vs {home_count})")
    else:
        print(f"⚖️  EVENLY MATCHED ({home_count} vs {away_count})")

    # Game Metrics (if week provided)
    if week:
        print("\n🏈 GAME PREDICTION METRICS")
        print("-" * 70)

        game_metrics = api.get_game_metrics(home_team, away_team, week=week)

        if 'summary' in game_metrics:
            summary = game_metrics['summary']

            # Pace
            print("\nPace:")
            pace = summary['pace']
            combined = pace['combined_pace']
            vs_avg = pace['pace_vs_league_avg']

            print(f"  {home_team}: {pace['home_plays_per_game']:.1f} plays/game")
            print(f"  {away_team}: {pace['away_plays_per_game']:.1f} plays/game")
            print(f"  Combined: {combined:.1f} (league avg: 65)")

            if vs_avg > 5:
                print(f"  → Expected FAST-paced game (+{vs_avg:.1f} plays)")
                print(f"  → Total adjustment: +{vs_avg / 10 * 3.5:.1f} points")
            elif vs_avg < -5:
                print(f"  → Expected SLOW-paced game ({vs_avg:.1f} plays)")
                print(f"  → Total adjustment: {vs_avg / 10 * 3.5:.1f} points")
            else:
                print(f"  → Expected AVERAGE pace")

            # Turnovers
            print("\nTurnover Margin:")
            to = summary['turnovers']
            margin_diff = to['margin_differential']

            print(f"  {home_team}: {to['home_margin']:+d}")
            print(f"  {away_team}: {to['away_margin']:+d}")
            print(f"  Differential: {margin_diff:+d}")

            if abs(margin_diff) >= 3:
                spread_impact = margin_diff * 2.5
                print(f"  → Significant TO edge: {spread_impact:+.1f} point spread impact")

            # Efficiency
            print("\nEfficiency:")
            eff = summary['efficiency']
            print(f"  {home_team} Success Rate: {eff['home_success_rate_off']:.1%}")
            print(f"  {away_team} Success Rate: {eff['away_success_rate_off']:.1%}")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare NFL team matchup")
    parser.add_argument("home_team", help="Home team abbreviation")
    parser.add_argument("away_team", help="Away team abbreviation")
    parser.add_argument("--week", type=int, help="Week number for game metrics")

    args = parser.parse_args()

    try:
        compare_matchup(args.home_team, args.away_team, week=args.week)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
