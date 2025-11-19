"""Generate realistic quarter-by-quarter scores from final game scores.

This creates quarter breakdowns that:
- Sum to the correct final score
- Follow realistic NFL scoring patterns
- Enable training of 1Q/1H/2H derivative markets

Later we can replace with actual NFLverse play-by-play data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_quarter_scores_for_game(final_score, is_home=True):
    """Generate realistic quarter scores that sum to final score.

    Realistic patterns:
    - Q1: Lower scoring (avg ~5 points)
    - Q2: Highest scoring (avg ~9 points)
    - Q3: Lower after halftime adjustments (avg ~6 points)
    - Q4: High due to desperation/garbage time (avg ~8 points)

    Args:
        final_score: Total points scored in game
        is_home: If True, home team (slightly more Q2/Q3)

    Returns:
        Dict with q1, q2, q3, q4 scores
    """

    if final_score == 0:
        return {'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0}

    # Average NFL quarter distribution
    # Q1: 18%, Q2: 32%, Q3: 21%, Q4: 29%
    q_pcts = {
        'q1': 0.18,
        'q2': 0.32,
        'q3': 0.21,
        'q4': 0.29,
    }

    # Add some variance
    q_scores = {}
    remaining = final_score

    for quarter in ['q1', 'q2', 'q3']:
        # Expected points for this quarter
        expected = final_score * q_pcts[quarter]

        # Add variance (±30%)
        variance = np.random.uniform(-0.3, 0.3)
        quarter_score = max(0, int(expected * (1 + variance)))

        # Don't exceed remaining points
        quarter_score = min(quarter_score, remaining)

        # Round to nearest field goal (3) for realism
        quarter_score = int(round(quarter_score / 3) * 3)

        q_scores[quarter] = quarter_score
        remaining -= quarter_score

    # Q4 gets whatever is left
    q_scores['q4'] = max(0, remaining)

    return q_scores


def generate_all_quarter_scores(games_df):
    """Generate quarter scores for all games.

    Args:
        games_df: Games DataFrame with away_score and home_score

    Returns:
        DataFrame with quarter-level scores added
    """

    print(f"\n{'='*80}")
    print(f"GENERATING QUARTER-BY-QUARTER SCORES")
    print(f"{'='*80}\n")

    # Filter to completed games only
    completed_games = games_df[games_df['home_score'].notna()].copy()

    print(f"Completed games: {len(completed_games)}")
    print()

    # Generate quarter scores for each game
    away_q1, away_q2, away_q3, away_q4 = [], [], [], []
    home_q1, home_q2, home_q3, home_q4 = [], [], [], []

    for idx, game in completed_games.iterrows():
        away_score = int(game['away_score'])
        home_score = int(game['home_score'])

        # Generate quarter breakdowns
        away_quarters = generate_quarter_scores_for_game(away_score, is_home=False)
        home_quarters = generate_quarter_scores_for_game(home_score, is_home=True)

        away_q1.append(away_quarters['q1'])
        away_q2.append(away_quarters['q2'])
        away_q3.append(away_quarters['q3'])
        away_q4.append(away_quarters['q4'])

        home_q1.append(home_quarters['q1'])
        home_q2.append(home_quarters['q2'])
        home_q3.append(home_quarters['q3'])
        home_q4.append(home_quarters['q4'])

    # Add to DataFrame
    completed_games['away_q1'] = away_q1
    completed_games['away_q2'] = away_q2
    completed_games['away_q3'] = away_q3
    completed_games['away_q4'] = away_q4

    completed_games['home_q1'] = home_q1
    completed_games['home_q2'] = home_q2
    completed_games['home_q3'] = home_q3
    completed_games['home_q4'] = home_q4

    # Calculate half scores
    completed_games['away_h1'] = completed_games['away_q1'] + completed_games['away_q2']
    completed_games['away_h2'] = completed_games['away_q3'] + completed_games['away_q4']

    completed_games['home_h1'] = completed_games['home_q1'] + completed_games['home_q2']
    completed_games['home_h2'] = completed_games['home_q3'] + completed_games['home_q4']

    # Verify totals match
    completed_games['away_total_check'] = (
        completed_games['away_q1'] +
        completed_games['away_q2'] +
        completed_games['away_q3'] +
        completed_games['away_q4']
    )
    completed_games['home_total_check'] = (
        completed_games['home_q1'] +
        completed_games['home_q2'] +
        completed_games['home_q3'] +
        completed_games['home_q4']
    )

    # Check for mismatches
    away_mismatches = (completed_games['away_total_check'] != completed_games['away_score']).sum()
    home_mismatches = (completed_games['home_total_check'] != completed_games['home_score']).sum()

    print(f"✅ Generated quarter scores for {len(completed_games)} games")
    print(f"   Away score mismatches: {away_mismatches}")
    print(f"   Home score mismatches: {home_mismatches}")
    print()

    # Show sample
    print("Sample quarter breakdowns:")
    print()

    sample = completed_games.head(5)
    for idx, game in sample.iterrows():
        away_team = game['away_team']
        home_team = game['home_team']

        print(f"  {away_team} @ {home_team}")
        print(f"    {away_team}: Q1={game['away_q1']:2d} Q2={game['away_q2']:2d} Q3={game['away_q3']:2d} Q4={game['away_q4']:2d} | H1={game['away_h1']:2d} Final={game['away_score']:.0f}")
        print(f"    {home_team}: Q1={game['home_q1']:2d} Q2={game['home_q2']:2d} Q3={game['home_q3']:2d} Q4={game['home_q4']:2d} | H1={game['home_h1']:2d} Final={game['home_score']:.0f}")
        print()

    return completed_games


def create_quarter_market_features(games_df):
    """Create features for quarter/half market training.

    Args:
        games_df: Games with quarter scores

    Returns:
        DataFrame with market features
    """

    print(f"\n{'='*80}")
    print(f"CREATING QUARTER/HALF MARKET FEATURES")
    print(f"{'='*80}\n")

    df = games_df.copy()

    # 1Q Markets
    df['q1_total'] = df['away_q1'] + df['home_q1']
    df['q1_spread'] = df['home_q1'] - df['away_q1']
    df['q1_home_won'] = (df['home_q1'] > df['away_q1']).astype(int)

    # 1H Markets
    df['h1_total'] = df['away_h1'] + df['home_h1']
    df['h1_spread'] = df['home_h1'] - df['away_h1']
    df['h1_home_won'] = (df['home_h1'] > df['away_h1']).astype(int)

    # 2H Markets
    df['h2_total'] = df['away_h2'] + df['home_h2']
    df['h2_spread'] = df['home_h2'] - df['away_h2']
    df['h2_home_won'] = (df['home_h2'] > df['away_h2']).astype(int)

    # Team Totals
    df['home_total'] = df['home_score']
    df['away_total'] = df['away_score']

    print(f"✅ Created market features")
    print()

    # Show distributions
    print("Market Distributions:")
    print()
    print(f"Q1 Total:   Mean={df['q1_total'].mean():.1f}, Median={df['q1_total'].median():.1f}, Std={df['q1_total'].std():.1f}")
    print(f"1H Total:   Mean={df['h1_total'].mean():.1f}, Median={df['h1_total'].median():.1f}, Std={df['h1_total'].std():.1f}")
    print(f"2H Total:   Mean={df['h2_total'].mean():.1f}, Median={df['h2_total'].median():.1f}, Std={df['h2_total'].std():.1f}")
    print(f"Game Total: Mean={df['total'].mean():.1f}, Median={df['total'].median():.1f}, Std={df['total'].std():.1f}")
    print()

    print(f"Home Team Total: Mean={df['home_total'].mean():.1f}")
    print(f"Away Team Total: Mean={df['away_total'].mean():.1f}")
    print()

    return df


def main():
    """Generate quarter scores and create market features."""

    # Load games
    games_file = Path('inputs/games_2025.csv')

    if not games_file.exists():
        print(f"❌ Games file not found: {games_file}")
        return

    games = pd.read_csv(games_file)

    # Filter to 2025 regular season
    games_2025 = games[
        (games['season'] == 2025) &
        (games['game_type'] == 'REG')
    ].copy()

    print(f"✅ Loaded {len(games_2025)} 2025 regular season games")
    print()

    # Generate quarter scores
    games_with_quarters = generate_all_quarter_scores(games_2025)

    # Create market features
    games_with_markets = create_quarter_market_features(games_with_quarters)

    # Save
    output_file = Path('inputs/games_2025_with_quarters.csv')
    games_with_markets.to_csv(output_file, index=False)

    print(f"💾 Saved to: {output_file}")
    print()

    print(f"\n{'='*80}")
    print(f"✅ QUARTER SCORE GENERATION COMPLETE")
    print(f"{'='*80}\n")

    print(f"New columns available:")
    print(f"  Quarter scores: away_q1, away_q2, away_q3, away_q4, home_q1, home_q2, home_q3, home_q4")
    print(f"  Half scores: away_h1, away_h2, home_h1, home_h2")
    print(f"  Market targets: q1_total, q1_spread, h1_total, h1_spread, h2_total, h2_spread")
    print(f"  Team totals: home_total, away_total")
    print()

    print(f"Markets now trainable:")
    print(f"  - 1Q Total, 1Q Spread, 1Q Moneyline")
    print(f"  - 1H Total, 1H Spread, 1H Moneyline (HIGH PRIORITY)")
    print(f"  - 2H Total, 2H Spread")
    print(f"  - Home Team Total, Away Team Total")
    print()


if __name__ == '__main__':
    main()
