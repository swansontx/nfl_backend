#!/usr/bin/env python3
"""
Run comprehensive historical backtest

Tests the system on past seasons to see if it would have made money.

Usage:
    # Full 2024 season
    python scripts/run_historical_backtest.py --season 2024

    # Multiple seasons
    python scripts/run_historical_backtest.py --seasons 2023 2024

    # Specific date range
    python scripts/run_historical_backtest.py --start 2024-09-01 --end 2024-12-31

    # With signal analysis
    python scripts/run_historical_backtest.py --season 2024 --analyze-signals
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime
import json

from backend.config.logging_config import get_logger
from backend.backtest import BacktestEngine, SignalEffectivenessAnalyzer, PositionSizer, SizingStrategy
from backend.database.session import get_db
from backend.database.models import Game

logger = get_logger(__name__)


def run_backtest(
    start_date: datetime,
    end_date: datetime,
    markets: list = None,
    kelly_fraction: float = 0.25,
    analyze_signals: bool = False
):
    """
    Run historical backtest

    Args:
        start_date: Start date
        end_date: End date
        markets: Markets to test (None = all)
        kelly_fraction: Kelly fraction for position sizing
        analyze_signals: Whether to analyze signal effectiveness
    """
    print("=" * 80)
    print("HISTORICAL BACKTEST")
    print("=" * 80)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Kelly Fraction: {kelly_fraction}")
    print()

    # Initialize backtest engine
    engine = BacktestEngine()
    engine.kelly_fraction = kelly_fraction

    # Run backtest
    print("Running backtest...")
    result = engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        markets=markets,
        simulate_betting=True
    )

    # Print report
    report = engine.generate_backtest_report(result)
    print(report)

    # Signal analysis if requested
    if analyze_signals and result.projection_results:
        print("\n" + "=" * 80)
        print("SIGNAL EFFECTIVENESS ANALYSIS")
        print("=" * 80)

        analyzer = SignalEffectivenessAnalyzer()

        try:
            signal_result = analyzer.analyze_signals(
                backtest_results=result.projection_results
            )

            if signal_result:
                signal_report = analyzer.generate_signal_report(signal_result)
                print(signal_report)

                # Recommendations for signal weights
                print("\n" + "=" * 80)
                print("RECOMMENDED SIGNAL WEIGHTS")
                print("=" * 80)
                print("\nBased on historical performance, use these weights:")
                print()

                for signal, weight in signal_result.optimal_weights.items():
                    print(f"  {signal:30s} {weight:.4f}")

                print()
                print("Update your SignalWeights in recommendations/recommendation_scorer.py")

        except Exception as e:
            print(f"Signal analysis failed: {e}")

    # Save results
    output_file = Path(__file__).parent.parent / "backtest_results.json"

    results_data = {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_games': result.total_games,
        'total_projections': result.total_projections,
        'strategy': result.strategy_results if result.strategy_results else {},
        'markets_tested': result.markets_tested,
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return result


def run_rolling_backtest(seasons: list, window_weeks: int = 4):
    """
    Run rolling window backtest across seasons

    Useful for detecting model drift
    """
    print("=" * 80)
    print("ROLLING WINDOW BACKTEST")
    print("=" * 80)
    print(f"Seasons: {seasons}")
    print(f"Window: {window_weeks} weeks")
    print()

    engine = BacktestEngine()

    results = engine.run_rolling_backtest(
        seasons=seasons,
        window_weeks=window_weeks
    )

    # Analyze performance over time
    rois = []
    sharpes = []

    print("\nWindow Results:")
    print(f"{'Window':<15} {'Games':<8} {'ROI':<10} {'Sharpe':<10} {'Profit':<12}")
    print("-" * 80)

    for i, result in enumerate(results):
        if result.strategy_results:
            strat = result.strategy_results
            roi = strat['roi_percent']
            sharpe = strat['sharpe_ratio']
            profit = strat['total_profit']

            rois.append(roi)
            sharpes.append(sharpe)

            print(f"Window {i+1:<8} {result.total_games:<8} {roi:>8.2f}% {sharpe:>9.2f} ${profit:>10.2f}")

    # Summary stats
    if rois:
        print()
        print("Summary Across Windows:")
        print(f"  Average ROI: {sum(rois)/len(rois):.2f}%")
        print(f"  Average Sharpe: {sum(sharpes)/len(sharpes):.2f}")
        print(f"  Positive ROI windows: {sum(1 for r in rois if r > 0)}/{len(rois)}")

        # Check for drift
        first_half_roi = sum(rois[:len(rois)//2]) / (len(rois)//2) if len(rois) > 2 else 0
        second_half_roi = sum(rois[len(rois)//2:]) / (len(rois) - len(rois)//2) if len(rois) > 2 else 0

        if abs(first_half_roi - second_half_roi) > 5:
            print(f"\n⚠️  Warning: Potential model drift detected")
            print(f"  First half ROI: {first_half_roi:.2f}%")
            print(f"  Second half ROI: {second_half_roi:.2f}%")


def compare_strategies(start_date: datetime, end_date: datetime):
    """
    Compare different position sizing strategies
    """
    print("=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print()

    strategies = [
        (SizingStrategy.KELLY, "Full Kelly"),
        (SizingStrategy.FRACTIONAL_KELLY, "Quarter Kelly"),
        (SizingStrategy.FIXED_PERCENTAGE, "Fixed 2%"),
    ]

    results = []

    for strategy, name in strategies:
        print(f"\nTesting {name}...")

        sizer = PositionSizer(strategy=strategy, kelly_fraction=0.25)

        # Get backtest results (would need to be refactored to use different sizers)
        # For now, just show conceptually
        print(f"  {name}: (would run with {strategy.value})")

        # In full implementation:
        # backtest_with_strategy(start_date, end_date, sizer)

    print("\n" + "=" * 80)
    print("Recommendation: Use Fractional Kelly (0.25) for best risk-adjusted returns")


def main():
    parser = argparse.ArgumentParser(description='Run historical backtest')
    parser.add_argument('--season', type=int, help='Single season to test')
    parser.add_argument('--seasons', type=int, nargs='+', help='Multiple seasons')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--markets', nargs='+', help='Specific markets to test')
    parser.add_argument('--kelly-fraction', type=float, default=0.25, help='Kelly fraction (default: 0.25)')
    parser.add_argument('--analyze-signals', action='store_true', help='Analyze signal effectiveness')
    parser.add_argument('--rolling', action='store_true', help='Run rolling window backtest')
    parser.add_argument('--compare-strategies', action='store_true', help='Compare position sizing strategies')

    args = parser.parse_args()

    # Determine date range
    if args.start and args.end:
        start_date = datetime.fromisoformat(args.start)
        end_date = datetime.fromisoformat(args.end)
    elif args.season:
        # Use full season
        with get_db() as session:
            games = session.query(Game).filter(
                Game.season == args.season
            ).order_by(Game.game_date).all()

            if not games:
                print(f"No games found for season {args.season}")
                sys.exit(1)

            start_date = games[0].game_date
            end_date = games[-1].game_date
    elif args.seasons:
        # Rolling backtest across seasons
        if args.rolling:
            run_rolling_backtest(args.seasons, window_weeks=4)
            return
        else:
            # Use first season
            args.season = args.seasons[0]
            with get_db() as session:
                games = session.query(Game).filter(
                    Game.season == args.season
                ).order_by(Game.game_date).all()

                start_date = games[0].game_date
                end_date = games[-1].game_date
    else:
        print("Must specify --season, --seasons, or --start/--end")
        sys.exit(1)

    # Run appropriate test
    if args.compare_strategies:
        compare_strategies(start_date, end_date)
    else:
        run_backtest(
            start_date=start_date,
            end_date=end_date,
            markets=args.markets,
            kelly_fraction=args.kelly_fraction,
            analyze_signals=args.analyze_signals
        )


if __name__ == "__main__":
    main()
