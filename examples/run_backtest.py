"""
Run game predictions backtest to validate metrics improvements.

This script backtests baseline vs enhanced predictions against
actual 2024 results to measure improvement from pace/turnover/efficiency metrics.

Usage:
    python examples/run_backtest.py
    python examples/run_backtest.py --weeks 1 2 3 4 5
    python examples/run_backtest.py --export backtest_results.csv
"""

import sys
import os
import argparse

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.backtesting.game_predictions_backtest import GamePredictionsBacktest


def main():
    parser = argparse.ArgumentParser(
        description="Backtest game predictions to validate metrics improvements"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2024,
        help="Season to backtest (default: 2024)"
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs='+',
        help="Specific weeks to test (default: all completed weeks)"
    )
    parser.add_argument(
        "--recent-weeks",
        type=int,
        default=4,
        help="Number of recent weeks to use for metrics (default: 4)"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to CSV file"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("GAME PREDICTIONS BACKTEST")
    print("="*70)
    print(f"\nSeason: {args.season}")
    if args.weeks:
        print(f"Weeks: {', '.join(map(str, args.weeks))}")
    else:
        print("Weeks: All completed weeks")
    print(f"Recent weeks for metrics: {args.recent_weeks}")

    # Initialize backtest engine
    print("\nInitializing backtest engine...")
    backtest = GamePredictionsBacktest(season=args.season)

    # Run backtest
    print("\nRunning backtest...")
    results = backtest.run_backtest(
        weeks=args.weeks,
        recent_weeks=args.recent_weeks
    )

    if not results:
        print("\n❌ No results to analyze. Check that games are completed and data is available.")
        return 1

    # Print report
    backtest.print_report(results)

    # Export if requested
    if args.export:
        backtest.export_results(results, args.export)
        print(f"\n✅ Results exported to {args.export}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
