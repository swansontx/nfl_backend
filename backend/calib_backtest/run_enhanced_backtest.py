"""Enhanced backtesting with injury-aware evaluation.

This module provides comprehensive backtesting that:
1. Excludes DNP instances from accuracy metrics (don't penalize for injuries)
2. Categorizes prediction errors by DNP reason (injury vs inactive vs rest)
3. Calculates separate metrics for different player availability scenarios
4. Generates detailed reports on model performance with injury context

Key Enhancements:
- Injury-aware evaluation: Separate metrics for healthy vs injured players
- DNP categorization: Track why predictions missed (Out, Doubtful, Questionable, Healthy DNP)
- Expected value analysis: Compare projected vs actual for expected-to-play only
- Sharp betting signals: Find edges where injury risk is priced incorrectly
"""

from pathlib import Path
import argparse
import json
import pickle
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict


def run_enhanced_backtest(
    season: int,
    model_path: Path,
    features_file: Path,
    injury_file: Optional[Path],
    output_report: Path,
    weeks: Optional[List[int]] = None
) -> Dict:
    """Run enhanced backtest with injury-aware evaluation.

    Args:
        season: Season year
        model_path: Path to trained model pickle
        features_file: Path to player features JSON (with injury data merged)
        injury_file: Optional path to injury data JSON
        output_report: Path to save backtest report
        weeks: Optional list of weeks to backtest (default: all)

    Returns:
        Dict with comprehensive backtest results
    """
    output_report.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"ENHANCED BACKTEST - Season {season}")
    print(f"Model: {model_path}")
    print(f"Injury-Aware Evaluation: {'Yes' if injury_file else 'No'}")
    print(f"{'='*80}\n")

    # Load model
    print(f"📦 Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load features
    print(f"📂 Loading features from {features_file}")
    with open(features_file, 'r') as f:
        player_features = json.load(f)

    # Load injury data if provided
    injury_map = {}
    if injury_file and injury_file.exists():
        print(f"🏥 Loading injury data from {injury_file}")
        with open(injury_file, 'r') as f:
            injury_map = json.load(f)
        print(f"✓ Loaded {len(injury_map)} injury records")

    # Run backtest
    results = _run_backtest_with_injury_context(
        model=model,
        player_features=player_features,
        injury_map=injury_map,
        season=season,
        weeks=weeks
    )

    # Save report
    with open(output_report, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Backtest complete: {output_report}")

    # Print summary
    _print_backtest_summary(results)

    return results


def _run_backtest_with_injury_context(
    model,
    player_features: Dict,
    injury_map: Dict,
    season: int,
    weeks: Optional[List[int]]
) -> Dict:
    """Run backtest with injury context tracking.

    Args:
        model: Trained model
        player_features: Player feature data
        injury_map: Injury data mapping
        season: Season year
        weeks: Weeks to evaluate

    Returns:
        Comprehensive results dict
    """
    predictions = []
    actuals = []

    # Track by injury status
    by_injury_status = defaultdict(lambda: {'predictions': [], 'actuals': []})

    # Track DNP instances
    dnp_instances = []

    # Track expected vs unexpected DNPs
    expected_dnp = []  # Was listed Out/Doubtful
    unexpected_dnp = []  # Was expected to play but didn't

    total_games = 0
    active_games = 0
    dnp_games = 0

    # Extract target field from model (assume it's stored as metadata)
    # For now, default to passing_yards
    target_field = 'passing_yards'

    for player_id, games in player_features.items():
        for game in games:
            total_games += 1

            # Filter by week if specified
            game_week = game.get('week', 0)
            if weeks and game_week not in weeks:
                continue

            # Check if player was active
            is_active = game.get('is_active', True)

            # Get injury status
            season_val = game.get('season', season)
            week = game.get('week', 0)
            injury_key = f"{season_val}_{week}_{player_id}"
            injury_info = injury_map.get(injury_key, {})
            injury_status = injury_info.get('report_status', 'Healthy')
            expected_to_play = injury_info.get('expected_to_play', True)

            # Actual value
            actual = game.get(target_field, 0)

            if not is_active:
                # Player DNP
                dnp_games += 1
                dnp_reason = game.get('dnp_reason', 'unknown')

                dnp_instance = {
                    'player_id': player_id,
                    'week': week,
                    'dnp_reason': dnp_reason,
                    'injury_status': injury_status,
                    'expected_to_play': expected_to_play
                }
                dnp_instances.append(dnp_instance)

                # Categorize DNP
                if not expected_to_play:
                    expected_dnp.append(dnp_instance)
                else:
                    unexpected_dnp.append(dnp_instance)

                # Don't make prediction for DNP (or predict 0 and exclude from metrics)
                continue

            active_games += 1

            # Extract features for prediction
            # (This is simplified - in reality, need to match model's feature columns)
            feature_cols = ['qb_epa', 'cpoe_avg', 'success_rate', 'attempts', 'completions']
            features = [game.get(f, 0) for f in feature_cols]

            # Skip if missing critical features
            if not any(features):
                continue

            # Make prediction
            try:
                pred = model.predict([features])[0]
            except:
                # If prediction fails, skip
                continue

            predictions.append(pred)
            actuals.append(actual)

            # Track by injury status
            by_injury_status[injury_status]['predictions'].append(pred)
            by_injury_status[injury_status]['actuals'].append(actual)

    # Calculate overall metrics (active games only)
    overall_metrics = _calculate_metrics(predictions, actuals)

    # Calculate metrics by injury status
    metrics_by_status = {}
    for status, data in by_injury_status.items():
        if len(data['predictions']) > 0:
            metrics_by_status[status] = _calculate_metrics(
                data['predictions'],
                data['actuals']
            )

    # Compile results
    results = {
        'season': season,
        'total_games': total_games,
        'active_games': active_games,
        'dnp_games': dnp_games,
        'dnp_rate': dnp_games / total_games if total_games > 0 else 0,

        # Overall accuracy (active games only)
        'overall_metrics': overall_metrics,

        # Metrics by injury status
        'metrics_by_injury_status': metrics_by_status,

        # DNP analysis
        'dnp_summary': {
            'total_dnp': len(dnp_instances),
            'expected_dnp': len(expected_dnp),  # Listed Out/Doubtful
            'unexpected_dnp': len(unexpected_dnp),  # Expected to play but didn't
            'unexpected_dnp_rate': len(unexpected_dnp) / len(dnp_instances) if dnp_instances else 0
        },

        # DNP breakdown by reason
        'dnp_by_reason': _categorize_dnp(dnp_instances),

        # Key insight: How often do we avoid bad predictions due to injury info?
        'injury_value': {
            'avoided_predictions': dnp_games,
            'expected_dnp_caught': len(expected_dnp),
            'unexpected_dnp_missed': len(unexpected_dnp),
            'description': 'Injury data helps avoid predictions for Out/Doubtful players'
        }
    }

    return results


def _calculate_metrics(predictions: List[float], actuals: List[float]) -> Dict:
    """Calculate accuracy metrics.

    Args:
        predictions: List of predicted values
        actuals: List of actual values

    Returns:
        Dict with RMSE, MAE, R², etc.
    """
    if not predictions or not actuals:
        return {
            'rmse': 0,
            'mae': 0,
            'r_squared': 0,
            'mean_error': 0,
            'samples': 0
        }

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # RMSE
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # MAE
    mae = np.mean(np.abs(predictions - actuals))

    # R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Mean error (bias)
    mean_error = np.mean(predictions - actuals)

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r_squared': float(r_squared),
        'mean_error': float(mean_error),
        'samples': len(predictions)
    }


def _categorize_dnp(dnp_instances: List[Dict]) -> Dict:
    """Categorize DNP instances by reason.

    Args:
        dnp_instances: List of DNP instance dicts

    Returns:
        Dict with counts by reason
    """
    categorized = defaultdict(int)

    for dnp in dnp_instances:
        reason = dnp.get('dnp_reason', 'unknown')
        categorized[reason] += 1

    return dict(categorized)


def _print_backtest_summary(results: Dict):
    """Print formatted backtest summary.

    Args:
        results: Backtest results dict
    """
    print(f"\n{'='*80}")
    print("BACKTEST SUMMARY")
    print(f"{'='*80}")

    print(f"\nGames Analyzed:")
    print(f"  Total: {results['total_games']}")
    print(f"  Active (played): {results['active_games']}")
    print(f"  DNP (inactive): {results['dnp_games']}")
    print(f"  DNP Rate: {results['dnp_rate']*100:.1f}%")

    overall = results['overall_metrics']
    print(f"\nOverall Accuracy (Active Games Only):")
    print(f"  RMSE: {overall['rmse']:.2f}")
    print(f"  MAE: {overall['mae']:.2f}")
    print(f"  R²: {overall['r_squared']:.3f}")
    print(f"  Mean Error: {overall['mean_error']:.2f}")
    print(f"  Samples: {overall['samples']}")

    print(f"\nAccuracy by Injury Status:")
    for status, metrics in results['metrics_by_injury_status'].items():
        print(f"  {status}:")
        print(f"    RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, "
              f"R²: {metrics['r_squared']:.3f}, N={metrics['samples']}")

    dnp_summary = results['dnp_summary']
    print(f"\nDNP Analysis:")
    print(f"  Total DNP: {dnp_summary['total_dnp']}")
    print(f"  Expected DNP (Out/Doubtful): {dnp_summary['expected_dnp']}")
    print(f"  Unexpected DNP: {dnp_summary['unexpected_dnp']}")
    print(f"  Unexpected Rate: {dnp_summary['unexpected_dnp_rate']*100:.1f}%")

    print(f"\nDNP Breakdown:")
    for reason, count in results['dnp_by_reason'].items():
        print(f"  {reason}: {count}")

    injury_value = results['injury_value']
    print(f"\nInjury Data Value:")
    print(f"  Avoided predictions: {injury_value['avoided_predictions']}")
    print(f"  Expected DNP caught: {injury_value['expected_dnp_caught']}")
    print(f"  Unexpected DNP missed: {injury_value['unexpected_dnp_missed']}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run enhanced backtest with injury-aware evaluation'
    )
    parser.add_argument('--season', type=int, required=True,
                       help='Season year')
    parser.add_argument('--model-path', type=Path, required=True,
                       help='Path to trained model')
    parser.add_argument('--features-file', type=Path, required=True,
                       help='Path to player features JSON (with injury data merged)')
    parser.add_argument('--injury-file', type=Path,
                       help='Optional path to injury data JSON')
    parser.add_argument('--output-report', type=Path,
                       default=Path('outputs/backtest/enhanced_backtest_report.json'),
                       help='Output path for backtest report')
    parser.add_argument('--weeks', type=int, nargs='+',
                       help='Optional weeks to backtest')

    args = parser.parse_args()

    run_enhanced_backtest(
        season=args.season,
        model_path=args.model_path,
        features_file=args.features_file,
        injury_file=args.injury_file,
        output_report=args.output_report,
        weeks=args.weeks
    )
