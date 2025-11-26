"""Test the game outcome orchestrator to understand its predictions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.orchestration.game_outcome_orchestrator import GameOutcomeOrchestrator, game_outcome_orchestrator
from backend.backtesting.framework import BacktestingFramework
import json


def test_orchestrator_on_sample_game():
    """Test orchestrator feature collection and prediction on a sample game."""

    print("\n" + "="*80)
    print("TESTING GAME OUTCOME ORCHESTRATOR")
    print("="*80 + "\n")

    # Load a sample game from historical data
    framework = BacktestingFramework()
    games_2023 = framework.load_historical_games(2023)

    if not games_2023:
        print("No historical games found")
        return

    # Get a sample game from week 10
    sample_game = None
    for game in games_2023:
        if game.week == 10:
            sample_game = game
            break

    if not sample_game:
        print("No sample game found")
        return

    print(f"Sample Game: {sample_game.away_team} @ {sample_game.home_team}")
    print(f"Week {sample_game.week}, {sample_game.season}")
    print(f"Actual Score: {sample_game.away_team} {sample_game.away_score}, {sample_game.home_team} {sample_game.home_score}")
    print(f"Actual Total: {sample_game.away_score + sample_game.home_score}")
    print(f"Actual Margin: {sample_game.home_score - sample_game.away_score}")
    print()

    # Create game_id in the format the orchestrator expects
    game_id = f"{sample_game.season}_{sample_game.week}_{sample_game.away_team}_{sample_game.home_team}"

    print(f"Game ID: {game_id}")
    print()

    # Initialize orchestrator
    orchestrator = GameOutcomeOrchestrator(season=sample_game.season)

    # Test 1: Collect features
    print("-" * 80)
    print("TEST 1: FEATURE COLLECTION")
    print("-" * 80 + "\n")

    try:
        features = orchestrator.collect_features(game_id, sample_game.week)

        print("Features collected:")
        print(f"  Home Team: {features.home_team}")
        print(f"  Away Team: {features.away_team}")
        print()

        print("Team Stats:")
        print(f"  Home Off PPG: {features.home_off_ppg:.1f}")
        print(f"  Home Def PPG: {features.home_def_ppg:.1f}")
        print(f"  Away Off PPG: {features.away_off_ppg:.1f}")
        print(f"  Away Def PPG: {features.away_def_ppg:.1f}")
        print()

        print("Advanced Metrics (EPA):")
        print(f"  Home Off EPA: {features.home_off_epa}")
        print(f"  Home Def EPA: {features.home_def_epa}")
        print(f"  Away Off EPA: {features.away_off_epa}")
        print(f"  Away Def EPA: {features.away_def_epa}")
        print()

        print("Recent Form (Last 3 Games):")
        print(f"  Home L3 Margin: {features.home_l3_margin:.1f}")
        print(f"  Away L3 Margin: {features.away_l3_margin:.1f}")
        print(f"  Home L3 Total: {features.home_l3_total:.1f}")
        print(f"  Away L3 Total: {features.away_l3_total:.1f}")
        print()

        print("Situational:")
        print(f"  Rest Differential: {features.rest_differential} days")
        print(f"  Division Game: {features.is_division_game}")
        print(f"  Primetime: {features.is_primetime}")
        print(f"  Dome: {features.is_dome}")
        print()

        print("Weather:")
        print(f"  Temperature: {features.temperature}°F" if features.temperature else "  Temperature: N/A")
        print(f"  Wind Speed: {features.wind_speed} mph" if features.wind_speed else "  Wind Speed: N/A")
        print(f"  Precipitation: {features.precipitation}" if features.precipitation else "  Precipitation: N/A")
        print()

        print("Market Data:")
        print(f"  Opening Spread: {features.opening_spread}" if features.opening_spread else "  Opening Spread: N/A")
        print(f"  Current Spread: {features.current_spread}" if features.current_spread else "  Current Spread: N/A")
        print(f"  Opening Total: {features.opening_total}" if features.opening_total else "  Opening Total: N/A")
        print(f"  Current Total: {features.current_total}" if features.current_total else "  Current Total: N/A")
        print()

        print("Public Betting:")
        print(f"  Spread Bet % (Home): {features.spread_bet_pct_home}" if features.spread_bet_pct_home else "  Spread Bet % (Home): N/A")
        print(f"  Spread Money % (Home): {features.spread_money_pct_home}" if features.spread_money_pct_home else "  Spread Money % (Home): N/A")
        print(f"  Total Bet % (Over): {features.total_bet_pct_over}" if features.total_bet_pct_over else "  Total Bet % (Over): N/A")
        print(f"  Total Money % (Over): {features.total_money_pct_over}" if features.total_money_pct_over else "  Total Money % (Over): N/A")
        print()

        print("Sharp Money Indicators:")
        print(f"  Sharp on Home: {features.spread_sharp_on_home}")
        print(f"  Sharp on Away: {features.spread_sharp_on_away}")
        print(f"  Sharp on Over: {features.total_sharp_on_over}")
        print(f"  Sharp on Under: {features.total_sharp_on_under}")
        print()

    except Exception as e:
        print(f"❌ Feature collection failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Generate prediction
    print("-" * 80)
    print("TEST 2: PREDICTION GENERATION")
    print("-" * 80 + "\n")

    try:
        prediction = orchestrator.predict_game(
            game_id=game_id,
            week=sample_game.week,
            market_spread=None,  # Don't provide market for now
            market_total=None
        )

        print("Orchestrator Prediction:")
        print(f"  Predicted Home Score: {prediction.predicted_home_score}")
        print(f"  Predicted Away Score: {prediction.predicted_away_score}")
        print(f"  Predicted Total: {prediction.predicted_total}")
        print(f"  Predicted Margin: {prediction.predicted_margin} (+ = home favored)")
        print(f"  Home Win Probability: {prediction.home_win_prob:.1%}")
        print()

        print("Uncertainty:")
        print(f"  Total Std Dev: ±{prediction.total_std:.1f}")
        print(f"  Total 95% CI: ({prediction.total_ci[0]:.1f}, {prediction.total_ci[1]:.1f})")
        print(f"  Margin Std Dev: ±{prediction.margin_std:.1f}")
        print(f"  Margin 95% CI: ({prediction.margin_ci[0]:.1f}, {prediction.margin_ci[1]:.1f})")
        print()

        print("Model Confidence:")
        print(f"  Confidence: {prediction.confidence:.1%}")
        print()

        # Compare to actual
        actual_total = sample_game.home_score + sample_game.away_score
        actual_margin = sample_game.home_score - sample_game.away_score

        total_error = abs(prediction.predicted_total - actual_total)
        margin_error = abs(prediction.predicted_margin - actual_margin)

        print("Accuracy vs Actual:")
        print(f"  Total Error: {total_error:.1f} points")
        print(f"  Margin Error: {margin_error:.1f} points")
        print()

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 3: Compare to simple baseline
    print("-" * 80)
    print("TEST 3: COMPARISON TO SIMPLE BASELINE")
    print("-" * 80 + "\n")

    # Simple baseline: recent team scoring averages
    import numpy as np

    # Get recent games for each team
    home_recent = [g for g in games_2023 if (g.home_team == sample_game.home_team or g.away_team == sample_game.home_team) and g.week < sample_game.week and g.week >= max(1, sample_game.week - 4)]
    away_recent = [g for g in games_2023 if (g.home_team == sample_game.away_team or g.away_team == sample_game.away_team) and g.week < sample_game.week and g.week >= max(1, sample_game.week - 4)]

    if home_recent:
        home_scores = [g.home_score if g.home_team == sample_game.home_team else g.away_score for g in home_recent]
        home_baseline = np.mean(home_scores)
    else:
        home_baseline = 22.0

    if away_recent:
        away_scores = [g.home_score if g.home_team == sample_game.away_team else g.away_score for g in away_recent]
        away_baseline = np.mean(away_scores)
    else:
        away_baseline = 22.0

    baseline_total = home_baseline + away_baseline
    baseline_margin = home_baseline - away_baseline

    baseline_total_error = abs(baseline_total - actual_total)
    baseline_margin_error = abs(baseline_margin - actual_margin)

    print("Simple Baseline (Recent Avg):")
    print(f"  Predicted Home Score: {home_baseline:.1f}")
    print(f"  Predicted Away Score: {away_baseline:.1f}")
    print(f"  Predicted Total: {baseline_total:.1f}")
    print(f"  Predicted Margin: {baseline_margin:.1f}")
    print()

    print("Baseline Accuracy:")
    print(f"  Total Error: {baseline_total_error:.1f} points")
    print(f"  Margin Error: {baseline_margin_error:.1f} points")
    print()

    print("-" * 80)
    print("WINNER:")
    print("-" * 80 + "\n")

    if total_error < baseline_total_error:
        print(f"✅ Orchestrator wins on total prediction by {baseline_total_error - total_error:.1f} points")
    elif total_error > baseline_total_error:
        print(f"❌ Baseline wins on total prediction by {total_error - baseline_total_error:.1f} points")
    else:
        print("🤝 Tie on total prediction")

    if margin_error < baseline_margin_error:
        print(f"✅ Orchestrator wins on margin prediction by {baseline_margin_error - margin_error:.1f} points")
    elif margin_error > baseline_margin_error:
        print(f"❌ Baseline wins on margin prediction by {margin_error - baseline_margin_error:.1f} points")
    else:
        print("🤝 Tie on margin prediction")

    print()


if __name__ == '__main__':
    test_orchestrator_on_sample_game()
