"""Generate synthetic kicker stats from game scores.

Kicker stats depend on team scoring patterns:
- XP Made = Team TDs (roughly)
- FG Made = Team drives that stalled in field goal range
- Total Kicker Points = (3 * FG Made) + XP Made

Realistic patterns:
- High scoring teams: More XPs, fewer FGs
- Low scoring teams: Fewer XPs, more FGs
- Team TDs ≈ Total Points / 7 (on average)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def estimate_team_kicker_stats(team_score):
    """Estimate kicker stats from team score.

    Args:
        team_score: Total points scored by team

    Returns:
        Dict with fg_made, xp_made, total_points
    """

    if team_score == 0:
        return {'fg_made': 0, 'xp_made': 0, 'total_points': 0}

    # Estimate TDs (avg 1 TD per 7-8 points)
    estimated_tds = team_score / 7.5

    # Add variance
    estimated_tds = max(0, estimated_tds + np.random.normal(0, 0.5))

    # XPs = TDs (with 98% make rate)
    xp_attempts = int(estimated_tds)
    xp_made = xp_attempts if np.random.random() < 0.98 else max(0, xp_attempts - 1)

    # Calculate points from TDs
    td_points = xp_made * 7

    # Remaining points must come from FGs
    remaining_points = team_score - td_points

    # FGs = remaining points / 3 (ensure non-negative)
    fg_made = max(0, int(remaining_points / 3))

    # Total kicker points
    total_points = max(0, (fg_made * 3) + xp_made)

    return {
        'fg_made': fg_made,
        'xp_made': xp_made,
        'total_points': total_points
    }


def generate_kicker_stats_from_games():
    """Generate kicker stats for all games."""

    print(f"\n{'='*80}")
    print(f"GENERATING KICKER STATS FROM GAME SCORES")
    print(f"{'='*80}\n")

    # Load games
    games_file = Path('inputs/games_2025_with_quarters.csv')

    if not games_file.exists():
        print(f"❌ Games file not found: {games_file}")
        print("   Run: python -m backend.analysis.generate_quarter_scores")
        return

    games = pd.read_csv(games_file)

    # Filter to completed games
    completed = games[games['home_score'].notna()].copy()

    print(f"✅ Loaded {len(completed)} completed games")
    print()

    # Kickers by team (realistic names)
    kickers_by_team = {
        'BUF': 'Tyler Bass',
        'KC': 'Harrison Butker',
        'BAL': 'Justin Tucker',
        'DET': 'Riley Patterson',
        'SF': 'Jake Moody',
        'PHI': 'Jake Elliott',
        'DAL': 'Brandon Aubrey',
        'MIA': 'Jason Sanders',
        'CIN': 'Evan McPherson',
        'LAC': 'Cameron Dicker',
        'MIN': 'Will Reichard',
        'SEA': 'Jason Myers',
        'LA': 'Joshua Karty',
        'TB': 'Chase McLaughlin',
        'ATL': 'Younghoe Koo',
        'GB': 'Brandon McManus',
        'HOU': 'Ka\'imi Fairbairn',
        'IND': 'Matt Gay',
        'NO': 'Blake Grupe',
        'WAS': 'Austin Seibert',
        'DEN': 'Wil Lutz',
        'PIT': 'Chris Boswell',
        'ARI': 'Chad Ryland',
        'NYJ': 'Greg Zuerlein',
        'CHI': 'Cairo Santos',
        'CLE': 'Dustin Hopkins',
        'NYG': 'Graham Gano',
        'CAR': 'Eddy Pineiro',
        'TEN': 'Nick Folk',
        'NE': 'Joey Slye',
        'LV': 'Daniel Carlson',
        'JAX': 'Cam Little',
    }

    # Generate kicker stats
    all_kicker_stats = []

    for idx, game in completed.iterrows():
        game_id = game['game_id']
        week = game['week']
        season = game['season']

        # Away kicker
        away_team = game['away_team']
        away_score = int(game['away_score'])

        away_kicker = kickers_by_team.get(away_team, f"{away_team} K")
        away_stats = estimate_team_kicker_stats(away_score)

        all_kicker_stats.append({
            'kicker_id': f"k_{away_team.lower()}",
            'kicker_name': away_kicker,
            'team': away_team,
            'opponent': game['home_team'],
            'week': week,
            'season': season,
            'game_id': game_id,
            'is_home': 0,
            **away_stats
        })

        # Home kicker
        home_team = game['home_team']
        home_score = int(game['home_score'])

        home_kicker = kickers_by_team.get(home_team, f"{home_team} K")
        home_stats = estimate_team_kicker_stats(home_score)

        all_kicker_stats.append({
            'kicker_id': f"k_{home_team.lower()}",
            'kicker_name': home_kicker,
            'team': home_team,
            'opponent': away_team,
            'week': week,
            'season': season,
            'game_id': game_id,
            'is_home': 1,
            **home_stats
        })

    # Create DataFrame
    kicker_df = pd.DataFrame(all_kicker_stats)

    print(f"✅ Generated {len(kicker_df)} kicker-game records")
    print()

    # Show distributions
    print("Kicker stat distributions:")
    print(f"  FG Made: Mean={kicker_df['fg_made'].mean():.2f}, Median={kicker_df['fg_made'].median():.0f}, Max={kicker_df['fg_made'].max()}")
    print(f"  XP Made: Mean={kicker_df['xp_made'].mean():.2f}, Median={kicker_df['xp_made'].median():.0f}, Max={kicker_df['xp_made'].max()}")
    print(f"  Total Points: Mean={kicker_df['total_points'].mean():.2f}, Median={kicker_df['total_points'].median():.0f}, Max={kicker_df['total_points'].max()}")
    print()

    # Save
    output_file = Path('inputs/kicker_stats_2025_synthetic.csv')
    kicker_df.to_csv(output_file, index=False)

    print(f"💾 Saved to: {output_file}")
    print()

    # Show samples
    print("Sample kicker performances:")
    print()

    # Top FG games
    top_fg = kicker_df.nlargest(5, 'fg_made')
    print("Most FGs made in a game:")
    for idx, row in top_fg.iterrows():
        print(f"  {row['kicker_name']:25s} (Week {row['week']:2d}): {row['fg_made']} FGs, {row['xp_made']} XPs, {row['total_points']} pts")
    print()

    # Top scoring games
    top_pts = kicker_df.nlargest(5, 'total_points')
    print("Most kicker points in a game:")
    for idx, row in top_pts.iterrows():
        print(f"  {row['kicker_name']:25s} (Week {row['week']:2d}): {row['fg_made']} FGs, {row['xp_made']} XPs, {row['total_points']} pts")
    print()

    return kicker_df


def main():
    """Generate kicker stats."""

    kicker_df = generate_kicker_stats_from_games()

    print(f"\n{'='*80}")
    print(f"✅ KICKER STATS GENERATION COMPLETE")
    print(f"{'='*80}\n")

    print("Next: Train kicker prop models (FG Made, XP Made, Total Points)")
    print()


if __name__ == '__main__':
    main()
