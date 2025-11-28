#!/usr/bin/env python3
"""
Run Signal Weight Optimization

Learns optimal weights for all prediction signals from historical data.
Uses Ridge regression with cross-validation.

NO HARDCODED ESTIMATES - everything learned from data!
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.features.signal_optimizer import learn_optimal_weights


def main():
    print("=" * 80)
    print("SIGNAL WEIGHT OPTIMIZATION")
    print("=" * 80)
    print("\nLearning optimal weights from 2025 historical data...")
    print("Using Ridge regression with 5-fold cross-validation")
    print("\n" + "=" * 80)

    # Learn optimal weights
    weights = learn_optimal_weights(season=2025, inputs_dir="inputs")

    # Display final results
    print("\n" + "=" * 80)
    print("FINAL OPTIMIZED WEIGHTS")
    print("=" * 80)

    print("\n📊 SPREAD PREDICTION WEIGHTS:")
    print(f"  Turnover Margin:     {weights.turnover_margin_weight:+.4f} pts per margin point")
    print(f"  EPA Differential:    {weights.epa_differential_weight:+.4f} pts per EPA")
    print(f"  Success Rate:        {weights.success_rate_weight:+.4f} pts per %")
    print(f"  Red Zone:            {weights.red_zone_weight:+.4f} pts per %")
    print(f"  Rest Differential:   {weights.rest_diff_weight:+.4f} pts per day")

    print("\n🏈 TOTAL PREDICTION WEIGHTS:")
    print(f"  Pace (plays/game):   {weights.pace_weight:+.4f} pts per play")
    print(f"  Explosive Play Rate: {weights.explosive_play_weight:+.4f} pts per %")
    print(f"  Wind Speed:          {weights.wind_weight:+.4f} pts per mph")
    print(f"  Primetime:           {weights.primetime_weight:+.4f} pts")
    print(f"  Divisional:          {weights.divisional_weight:+.4f} pts")

    print("\n📈 PERFORMANCE METRICS:")
    print(f"  Spread MAE:          {weights.spread_mae:.2f} points")
    print(f"  Total MAE:           {weights.total_mae:.2f} points")
    print(f"  Spread ATS Win %:    {weights.spread_ats_pct:.1f}%")
    print(f"  Total O/U Win %:     {weights.total_ou_pct:.1f}%")

    print("\n" + "=" * 80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print("\nThese weights are learned from data, not estimated.")
    print("Next step: Replace hardcoded multipliers with these learned values.")
    print()


if __name__ == "__main__":
    main()
