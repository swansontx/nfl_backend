#!/usr/bin/env python3
"""
Backtest Learned Weights vs Current Best

Compares Ridge-learned weights against our current calibrated approach.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from backend.features.signal_optimizer import SignalWeightOptimizer


def apply_learned_weights_predictions(optimizer: SignalWeightOptimizer, weights: dict) -> pd.DataFrame:
    """
    Generate predictions using learned weights.

    Returns DataFrame with game_id, predicted_spread, predicted_total.
    """
    results = []

    for idx, game in optimizer.games_df.iterrows():
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        week = game['week']

        # Extract signals
        signals = optimizer.extract_all_signals(game_id, home_team, away_team, week)

        # Add baselines (not included in extract_all_signals)
        baseline_spread = 0.0  # No hardcoded HFA
        baseline_total = 45.0  # League average

        # Apply learned spread weights
        spread_pred = (
            weights['spread'].get('baseline_spread', 1.0) * baseline_spread +
            weights['spread']['turnover_margin_diff'] * signals['turnover_margin_diff'] +
            weights['spread']['success_rate_diff'] * signals['success_rate_diff'] +
            weights['spread']['red_zone_diff'] * signals['red_zone_diff'] +
            weights['spread']['rest_differential'] * signals['rest_differential'] +
            weights['spread']['is_primetime'] * signals['is_primetime'] +
            weights['spread']['is_divisional'] * signals['is_divisional']
            # NOTE: Excluding EPA (negative weight)
        )

        # Apply learned total weights
        total_pred = (
            weights['total']['baseline_total'] * baseline_total +
            weights['total']['combined_pace'] * signals['combined_pace'] +
            weights['total']['combined_explosive'] * signals['combined_explosive'] +
            weights['total']['wind_speed'] * signals['wind_speed'] +
            weights['total']['temperature'] * signals['temperature'] +
            weights['total']['is_primetime'] * signals['is_primetime'] +
            weights['total']['is_divisional'] * signals['is_divisional'] +
            weights['total']['is_outdoor'] * signals['is_outdoor']
        )

        results.append({
            'game_id': game_id,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            'actual_spread': game['actual_spread'],
            'actual_total': game['actual_total'],
            'predicted_spread': spread_pred,
            'predicted_total': total_pred,
        })

    return pd.DataFrame(results)


def calculate_metrics(predictions_df: pd.DataFrame) -> dict:
    """Calculate performance metrics."""

    # Spread errors
    spread_errors = np.abs(predictions_df['actual_spread'] - predictions_df['predicted_spread'])
    spread_mae = spread_errors.mean()

    # Total errors
    total_errors = np.abs(predictions_df['actual_total'] - predictions_df['predicted_total'])
    total_mae = total_errors.mean()

    # ATS win rate (simplified - just direction)
    spread_correct = (
        ((predictions_df['predicted_spread'] > 0) & (predictions_df['actual_spread'] > 0)) |
        ((predictions_df['predicted_spread'] < 0) & (predictions_df['actual_spread'] < 0))
    )
    ats_win_pct = (spread_correct.sum() / len(predictions_df)) * 100

    # O/U win rate (simplified - just over/under)
    ou_correct = (
        ((predictions_df['predicted_total'] > 45) & (predictions_df['actual_total'] > 45)) |
        ((predictions_df['predicted_total'] < 45) & (predictions_df['actual_total'] < 45))
    )
    ou_win_pct = (ou_correct.sum() / len(predictions_df)) * 100

    return {
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'ats_win_pct': ats_win_pct,
        'ou_win_pct': ou_win_pct,
        'n_games': len(predictions_df),
    }


def main():
    print("=" * 80)
    print("BACKTEST: LEARNED WEIGHTS VALIDATION")
    print("=" * 80)

    # Load optimizer with all data
    print("\nLoading 2025 historical data...")
    optimizer = SignalWeightOptimizer(season=2025, inputs_dir="inputs")

    print(f"Games available: {len(optimizer.games_df)}")

    # Build feature matrix
    print("\nExtracting features...")
    features, spread_targets, total_targets = optimizer.build_feature_matrix()

    # Learn weights
    print("\nLearning optimal weights...")
    spread_weights = optimizer.optimize_spread_weights(features, spread_targets, cv_folds=5)
    total_weights = optimizer.optimize_total_weights(features, total_targets, cv_folds=5)

    # Package weights
    weights = {
        'spread': spread_weights,
        'total': total_weights,
    }

    # Generate predictions
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS WITH LEARNED WEIGHTS")
    print("=" * 80)

    predictions = apply_learned_weights_predictions(optimizer, weights)

    # Calculate metrics
    metrics = calculate_metrics(predictions)

    # Display results
    print("\n" + "=" * 80)
    print("LEARNED WEIGHTS PERFORMANCE")
    print("=" * 80)

    print(f"\n📊 SPREAD PREDICTIONS:")
    print(f"  MAE:         {metrics['spread_mae']:.2f} points")
    print(f"  ATS Win %:   {metrics['ats_win_pct']:.1f}%")

    print(f"\n🏈 TOTAL PREDICTIONS:")
    print(f"  MAE:         {metrics['total_mae']:.2f} points")
    print(f"  O/U Win %:   {metrics['ou_win_pct']:.1f}%")

    print(f"\n📈 SAMPLE SIZE:")
    print(f"  Games:       {metrics['n_games']}")

    print("\n" + "=" * 80)
    print("COMPARISON TO BENCHMARKS")
    print("=" * 80)

    # Compare to known benchmarks
    baseline_spread_mae = 11.45
    baseline_total_mae = 10.52
    current_spread_mae = 11.38
    current_total_mae = 10.63

    print(f"\n📊 SPREAD MAE:")
    print(f"  Baseline:         {baseline_spread_mae:.2f} pts")
    print(f"  Current Best:     {current_spread_mae:.2f} pts")
    print(f"  Learned Weights:  {metrics['spread_mae']:.2f} pts")
    spread_improvement = ((current_spread_mae - metrics['spread_mae']) / current_spread_mae) * 100
    print(f"  Improvement:      {spread_improvement:+.1f}%")

    print(f"\n🏈 TOTAL MAE:")
    print(f"  Baseline:         {baseline_total_mae:.2f} pts")
    print(f"  Current Best:     {current_total_mae:.2f} pts")
    print(f"  Learned Weights:  {metrics['total_mae']:.2f} pts")
    total_improvement = ((current_total_mae - metrics['total_mae']) / current_total_mae) * 100
    print(f"  Improvement:      {total_improvement:+.1f}%")

    # Save predictions
    output_file = "outputs/backtest_learned_weights.csv"
    predictions.to_csv(output_file, index=False)
    print(f"\n💾 Predictions saved to: {output_file}")

    print("\n" + "=" * 80)

    # Determine recommendation
    if metrics['spread_mae'] < current_spread_mae:
        print("✅ RECOMMENDATION: Deploy learned weights for SPREADS")
        print(f"   Expected improvement: {spread_improvement:.1f}%")
    else:
        print("⚠️  RECOMMENDATION: Keep current approach for spreads")

    if metrics['total_mae'] < current_total_mae:
        print("✅ RECOMMENDATION: Deploy learned weights for TOTALS")
        print(f"   Expected improvement: {total_improvement:.1f}%")
    else:
        print("⚠️  RECOMMENDATION: Keep current approach for totals")

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
