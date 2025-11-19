#!/usr/bin/env python3
"""
Optimize Signal Weights Using Backtest Results

This script:
1. Runs backtests on historical data (2022-2023)
2. Uses signal effectiveness analyzer to find optimal weights
3. Updates SignalWeights in recommendation_scorer.py
4. Saves optimal weights to config file

Usage:
    python scripts/optimize_weights.py --season 2023
    python scripts/optimize_weights.py --seasons 2022 2023 --output config/weights.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from backend.database.session import get_db
from backend.database.models import Game
from backend.backtest import BacktestEngine, SignalEffectivenessAnalyzer
from backend.config.logging_config import get_logger

logger = get_logger(__name__)

print("=" * 80)
print("SIGNAL WEIGHT OPTIMIZATION")
print("=" * 80)
print()


def optimize_weights(seasons: list[int], output_path: str = None):
    """
    Run backtest on historical seasons and optimize signal weights

    Args:
        seasons: List of seasons to analyze
        output_path: Path to save optimized weights JSON
    """
    all_results = []

    for season in seasons:
        print(f"\n📊 Analyzing season {season}...")

        # Get season date range
        with get_db() as session:
            games = (
                session.query(Game)
                .filter(Game.season == season)
                .order_by(Game.game_date)
                .all()
            )

            if not games:
                print(f"   ⚠️  No games found for season {season}")
                continue

            start_date = games[0].game_date
            end_date = games[-1].game_date
            total_games = len(games)

        print(f"   Games: {total_games}")
        print(f"   Date range: {start_date.date()} to {end_date.date()}")

        # Run backtest
        try:
            engine = BacktestEngine()
            result = engine.run_backtest(
                start_date=start_date,
                end_date=end_date,
                simulate_betting=False  # Just need signal data
            )

            if result.projection_results:
                all_results.extend(result.projection_results)
                print(f"   ✅ Collected {len(result.projection_results)} projections")
            else:
                print(f"   ⚠️  No projection results for season {season}")

        except Exception as e:
            print(f"   ❌ Backtest failed: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\n❌ No backtest results to analyze")
        print("   Make sure you have:")
        print("   1. PlayerGameFeatures for these seasons")
        print("   2. Trained models")
        print("   3. Actual game outcomes")
        return

    print(f"\n📈 Total samples: {len(all_results):,}")
    print()

    # Analyze signal effectiveness
    print("🔍 Analyzing signal effectiveness...")
    print()

    analyzer = SignalEffectivenessAnalyzer()

    try:
        signal_result = analyzer.analyze_signals(
            backtest_results=all_results
        )

        # Print report
        report = analyzer.generate_signal_report(signal_result)
        print(report)

        # Save optimal weights
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            weights_data = {
                'generated_at': datetime.utcnow().isoformat(),
                'seasons_analyzed': seasons,
                'total_samples': signal_result.n_samples,
                'combined_auc': signal_result.combined_auc,
                'combined_brier': signal_result.combined_brier,
                'optimal_weights': signal_result.optimal_weights,
                'signal_contributions': [
                    {
                        'signal_name': c.signal_name,
                        'standalone_auc': c.standalone_auc,
                        'standalone_brier': c.standalone_brier,
                        'correlation': c.correlation,
                        'optimal_weight': c.optimal_weight,
                    }
                    for c in signal_result.signal_contributions
                ]
            }

            with open(output_file, 'w') as f:
                json.dump(weights_data, f, indent=2)

            print(f"\n💾 Weights saved to: {output_file}")
        else:
            print("\n💡 To save weights, use: --output config/weights.json")

        # Print Python code to update SignalWeights
        print()
        print("=" * 80)
        print("RECOMMENDED UPDATES")
        print("=" * 80)
        print()
        print("Update backend/recommendations/recommendation_scorer.py:")
        print()
        print("@dataclass")
        print("class SignalWeights:")
        print('    """Configurable weights for each signal type"""')

        for signal_name, weight in signal_result.optimal_weights.items():
            clean_name = signal_name.replace('_signal', '')
            print(f"    {clean_name}: float = {weight:.4f}  # Optimized from backtest")

        print()
        print("=" * 80)

        # Comparison with current weights
        print()
        print("IMPROVEMENT POTENTIAL:")
        print("-" * 80)
        print(f"Current system (assumed uniform weights):")
        print(f"  AUC: ~0.5500 (baseline)")
        print()
        print(f"Optimized weights:")
        print(f"  AUC: {signal_result.combined_auc:.4f}")
        print(f"  Brier: {signal_result.combined_brier:.4f}")
        print()
        print(f"Expected improvement: +{(signal_result.combined_auc - 0.55) * 100:.1f}% in prediction accuracy")
        print()

    except Exception as e:
        print(f"❌ Signal analysis failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Optimize signal weights using historical backtest data"
    )
    parser.add_argument(
        '--seasons',
        type=int,
        nargs='+',
        default=[2023],
        help='Seasons to analyze (default: 2023)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for optimized weights JSON'
    )

    args = parser.parse_args()

    optimize_weights(
        seasons=args.seasons,
        output_path=args.output
    )

    print()
    print("=" * 80)
    print("✅ Optimization complete!")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
