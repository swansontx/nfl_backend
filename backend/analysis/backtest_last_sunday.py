"""Backtest predictions for last Sunday's games.

This script:
1. Identifies games from last Sunday (2025-11-17)
2. Makes predictions using only pre-game data
3. Generates prop recommendations
4. Fetches actual results
5. Compares predictions vs actuals

Usage:
    python -m backend.analysis.backtest_last_sunday
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json


def find_last_sunday_games(schedule_file: Path, target_date: str = "2025-11-17") -> pd.DataFrame:
    """Find all games played on last Sunday.

    Args:
        schedule_file: Path to schedule CSV
        target_date: Target date in YYYY-MM-DD format

    Returns:
        DataFrame of games on that date
    """
    print(f"\n{'='*80}")
    print(f"STEP 1: IDENTIFYING LAST SUNDAY'S GAMES ({target_date})")
    print(f"{'='*80}\n")

    if not schedule_file.exists():
        print(f"❌ Schedule file not found: {schedule_file}")
        print(f"   Run: python -m backend.ingestion.fetch_nflverse_schedules --year 2025")
        return pd.DataFrame()

    schedule = pd.read_csv(schedule_file)

    # Filter to target date
    schedule['gameday'] = pd.to_datetime(schedule['gameday']).dt.strftime('%Y-%m-%d')
    sunday_games = schedule[schedule['gameday'] == target_date].copy()

    print(f"📅 Found {len(sunday_games)} games on {target_date}:")
    print()

    for idx, game in sunday_games.iterrows():
        game_time = game.get('gametime', 'N/A')
        away = game['away_team']
        home = game['home_team']
        away_score = game.get('away_score', '?')
        home_score = game.get('home_score', '?')

        print(f"  {away} @ {home}")
        print(f"    Time: {game_time}")
        print(f"    Final: {away} {away_score}, {home} {home_score}")
        print()

    return sunday_games


def make_game_predictions(games_df: pd.DataFrame) -> pd.DataFrame:
    """Make pre-game predictions (spread, total, winner).

    Args:
        games_df: DataFrame of games to predict

    Returns:
        DataFrame with predictions added
    """
    print(f"\n{'='*80}")
    print(f"STEP 2: MAKING GAME PREDICTIONS (NO CHEATING - PRE-GAME ONLY)")
    print(f"{'='*80}\n")

    predictions = []

    for idx, game in games_df.iterrows():
        game_id = game['game_id']
        away_team = game['away_team']
        home_team = game['home_team']

        # Use Vegas lines as baseline (these are pre-game)
        spread = game.get('spread_line', np.nan)
        total = game.get('total_line', np.nan)

        # Predict winner based on spread
        if pd.notna(spread):
            predicted_winner = home_team if spread < 0 else away_team
            home_win_prob = _spread_to_win_probability(spread)
        else:
            predicted_winner = "UNKNOWN"
            home_win_prob = 0.5

        # Predict score based on total and spread
        if pd.notna(total) and pd.notna(spread):
            # Split total based on spread
            home_implied = (total - spread) / 2
            away_implied = (total + spread) / 2

            predicted_home_score = round(home_implied)
            predicted_away_score = round(away_implied)
        else:
            predicted_home_score = None
            predicted_away_score = None

        predictions.append({
            'game_id': game_id,
            'away_team': away_team,
            'home_team': home_team,
            'predicted_winner': predicted_winner,
            'home_win_prob': home_win_prob,
            'predicted_spread': spread,
            'predicted_total': total,
            'predicted_home_score': predicted_home_score,
            'predicted_away_score': predicted_away_score,
            'actual_home_score': game.get('home_score', None),
            'actual_away_score': game.get('away_score', None),
        })

        print(f"🏈 {away_team} @ {home_team}")
        print(f"   Predicted: {away_team} {predicted_away_score}, {home_team} {predicted_home_score}")
        print(f"   Winner: {predicted_winner} ({home_win_prob:.1%} home win prob)")
        print(f"   Spread: {spread:.1f}, Total: {total:.1f}")
        print()

    return pd.DataFrame(predictions)


def _spread_to_win_probability(spread: float) -> float:
    """Convert point spread to win probability.

    Rule of thumb: Each point is worth ~2.5% win probability around 50/50.
    Home favorite by 7 points ≈ 67% win probability.

    Args:
        spread: Point spread (negative = home favored)

    Returns:
        Home team win probability (0-1)
    """
    # Logistic function centered at 50%
    # spread = -7 (home favored by 7) → ~67%
    # spread = +7 (away favored by 7) → ~33%
    prob = 1 / (1 + np.exp(spread / 4))  # 4 points ≈ 10% prob shift
    return prob


def generate_prop_recommendations(games_df: pd.DataFrame) -> List[Dict]:
    """Generate prop recommendations for games.

    In a real scenario, this would:
    1. Load historical player stats (up to but NOT including target date)
    2. Train models on that historical data
    3. Make predictions for each player prop
    4. Compare to market lines
    5. Identify +EV opportunities

    For now, we'll simulate this process.

    Args:
        games_df: DataFrame of games

    Returns:
        List of prop recommendations
    """
    print(f"\n{'='*80}")
    print(f"STEP 3: GENERATING PROP RECOMMENDATIONS")
    print(f"{'='*80}\n")

    print("⚠️  NOTE: Full prop prediction requires:")
    print("   - Historical play-by-play data (up to 2025-11-16)")
    print("   - Trained models on that historical data")
    print("   - Market prop lines (would fetch from odds API)")
    print()
    print("   For this demo, we'll show the PROCESS but use simulated data")
    print()

    # Check if we have the infrastructure
    models_dir = Path('outputs/models')
    if not models_dir.exists():
        print("❌ No trained models found in outputs/models/")
        print("   Would need to train models on historical data first")
        return []

    # Simulated prop recommendations
    props = []

    for idx, game in games_df.iterrows():
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']

        # Example props (in real scenario, would predict all players)
        example_props = [
            {
                'game_id': game_id,
                'player': f'{home_team} QB',
                'prop_type': 'pass_yards',
                'line': 265.5,
                'prediction': 275.3,
                'edge': 0.068,
                'ev': 0.042,
                'recommendation': 'OVER',
                'confidence': 0.72,
            },
            {
                'game_id': game_id,
                'player': f'{home_team} RB1',
                'prop_type': 'rush_yards',
                'line': 72.5,
                'prediction': 68.2,
                'edge': -0.059,
                'ev': -0.031,
                'recommendation': 'PASS',
                'confidence': 0.58,
            },
        ]

        props.extend(example_props)

    # Show recommendations
    recommendations = [p for p in props if p['recommendation'] != 'PASS']

    print(f"💰 Generated {len(recommendations)} prop recommendations:")
    print()

    for prop in recommendations:
        print(f"  ✅ {prop['player']} {prop['prop_type']} {prop['recommendation']} {prop['line']}")
        print(f"     Prediction: {prop['prediction']:.1f} | Edge: {prop['edge']:+.1%} | EV: {prop['ev']:+.1%}")
        print()

    return recommendations


def load_actual_results(games_df: pd.DataFrame) -> pd.DataFrame:
    """Load actual game results and player stats.

    Args:
        games_df: DataFrame of games

    Returns:
        DataFrame with actual results
    """
    print(f"\n{'='*80}")
    print(f"STEP 4: LOADING ACTUAL RESULTS")
    print(f"{'='*80}\n")

    # Results are already in the schedule file
    results = []

    for idx, game in games_df.iterrows():
        game_id = game['game_id']
        away_team = game['away_team']
        home_team = game['home_team']
        away_score = game.get('away_score', None)
        home_score = game.get('home_score', None)

        if pd.isna(away_score) or pd.isna(home_score):
            print(f"⚠️  {away_team} @ {home_team}: NO RESULTS YET")
            actual_winner = None
            actual_total = None
            actual_spread = None
        else:
            actual_winner = home_team if home_score > away_score else away_team
            actual_total = away_score + home_score
            actual_spread = home_score - away_score  # negative = away won by more

            print(f"✅ {away_team} {int(away_score)} @ {home_team} {int(home_score)}")
            print(f"   Winner: {actual_winner}")
            print(f"   Total: {actual_total} | Spread: {actual_spread:+.0f}")
            print()

        results.append({
            'game_id': game_id,
            'away_team': away_team,
            'home_team': home_team,
            'actual_away_score': away_score,
            'actual_home_score': home_score,
            'actual_winner': actual_winner,
            'actual_total': actual_total,
            'actual_spread': actual_spread,
        })

    return pd.DataFrame(results)


def compare_predictions_vs_actuals(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame) -> Dict:
    """Compare predictions to actual results.

    Args:
        predictions_df: DataFrame of predictions
        actuals_df: DataFrame of actual results

    Returns:
        Dictionary of performance metrics
    """
    print(f"\n{'='*80}")
    print(f"STEP 5: COMPARING PREDICTIONS VS ACTUALS")
    print(f"{'='*80}\n")

    # Merge predictions and actuals
    merged = predictions_df.merge(
        actuals_df,
        on=['game_id', 'away_team', 'home_team'],
        how='left'
    )

    # Filter to games with results
    completed = merged[merged['actual_winner'].notna()].copy()

    if len(completed) == 0:
        print("❌ No completed games found (games may not have been played yet)")
        return {}

    # Calculate accuracy metrics
    winner_correct = (completed['predicted_winner'] == completed['actual_winner']).sum()
    winner_accuracy = winner_correct / len(completed)

    # Spread accuracy (within 3 points)
    spread_errors = abs(completed['predicted_spread'] - completed['actual_spread'])
    spread_mae = spread_errors.mean()
    spread_within_3 = (spread_errors <= 3).sum() / len(completed)

    # Total accuracy (within 3 points)
    total_errors = abs(completed['predicted_total'] - completed['actual_total'])
    total_mae = total_errors.mean()
    total_within_3 = (total_errors <= 3).sum() / len(completed)

    # Print results
    print(f"📊 GAME PREDICTION PERFORMANCE:")
    print(f"   Games analyzed: {len(completed)}")
    print()
    print(f"   Winner accuracy: {winner_correct}/{len(completed)} ({winner_accuracy:.1%})")
    print(f"   Spread MAE: {spread_mae:.2f} points")
    print(f"   Spread within 3: {spread_within_3:.1%}")
    print(f"   Total MAE: {total_mae:.2f} points")
    print(f"   Total within 3: {total_within_3:.1%}")
    print()

    # Detailed game-by-game
    print(f"📋 GAME-BY-GAME RESULTS:")
    print()

    for idx, game in completed.iterrows():
        away = game['away_team']
        home = game['home_team']

        pred_winner = game['predicted_winner']
        actual_winner = game['actual_winner']
        winner_correct = "✅" if pred_winner == actual_winner else "❌"

        pred_spread = game['predicted_spread']
        actual_spread = game['actual_spread']
        spread_error = abs(pred_spread - actual_spread)
        spread_status = "✅" if spread_error <= 3 else "❌"

        pred_total = game['predicted_total']
        actual_total = game['actual_total']
        total_error = abs(pred_total - actual_total)
        total_status = "✅" if total_error <= 3 else "❌"

        print(f"  {away} @ {home}")
        print(f"    Predicted: {away} {game['predicted_away_score']}, {home} {game['predicted_home_score']}")
        print(f"    Actual:    {away} {int(game['actual_away_score'])}, {home} {int(game['actual_home_score'])}")
        print(f"    Winner: {winner_correct} (predicted {pred_winner}, was {actual_winner})")
        print(f"    Spread: {spread_status} (off by {spread_error:.1f} points)")
        print(f"    Total:  {total_status} (off by {total_error:.1f} points)")
        print()

    return {
        'games_analyzed': len(completed),
        'winner_accuracy': winner_accuracy,
        'spread_mae': spread_mae,
        'spread_within_3': spread_within_3,
        'total_mae': total_mae,
        'total_within_3': total_within_3,
    }


def run_backtest(target_date: str = "2025-11-17"):
    """Run full backtest for last Sunday's games.

    Args:
        target_date: Date to backtest (YYYY-MM-DD)
    """
    print(f"\n{'#'*80}")
    print(f"# BACKTESTING LAST SUNDAY'S GAMES ({target_date})")
    print(f"{'#'*80}\n")

    # Setup paths
    inputs_dir = Path('inputs')
    outputs_dir = Path('outputs')

    # Determine season year (2025 for Nov 2025 game)
    season_year = int(target_date.split('-')[0])
    schedule_file = inputs_dir / f'schedule_{season_year}.csv'

    # Step 1: Find games
    games = find_last_sunday_games(schedule_file, target_date)

    if len(games) == 0:
        print("\n❌ No games found. Exiting.")
        return

    # Step 2: Make predictions
    predictions = make_game_predictions(games)

    # Step 3: Generate prop recommendations
    prop_recs = generate_prop_recommendations(games)

    # Step 4: Load actual results
    actuals = load_actual_results(games)

    # Step 5: Compare
    metrics = compare_predictions_vs_actuals(predictions, actuals)

    # Save results
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_file = outputs_dir / f'backtest_{target_date}.json'

    report = {
        'target_date': target_date,
        'games_count': len(games),
        'predictions': predictions.to_dict(orient='records'),
        'prop_recommendations': prop_recs,
        'actuals': actuals.to_dict(orient='records'),
        'metrics': metrics,
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ BACKTEST COMPLETE")
    print(f"{'='*80}\n")
    print(f"📄 Report saved to: {report_file}")
    print()


if __name__ == '__main__':
    import sys

    # Allow custom date via command line
    target_date = sys.argv[1] if len(sys.argv) > 1 else "2025-11-17"

    run_backtest(target_date)
