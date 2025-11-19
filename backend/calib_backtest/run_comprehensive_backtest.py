"""Comprehensive multi-prop backtesting pipeline.

This module runs backtests on ALL 60+ prop models and generates:
1. Performance metrics for each prop type (RMSE, MAE, R²)
2. Model rankings by accuracy
3. Identification of best/worst performing markets
4. Prop type-specific insights (which models are reliable for betting)
5. Comprehensive HTML/JSON reports

Use cases:
- Validate all models meet minimum accuracy thresholds before deployment
- Identify which prop types to focus betting on
- Find which models need retraining
- Generate confidence scores for each market
"""

from pathlib import Path
import argparse
import json
import pickle
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime


def run_comprehensive_backtest(
    season: int,
    models_dir: Path,
    features_file: Path,
    injury_file: Optional[Path],
    output_dir: Path,
    min_r2: float = 0.3,  # Minimum R² threshold
    min_samples: int = 10  # Minimum samples per model
) -> Dict:
    """Run comprehensive backtest on all multi-prop models.

    Args:
        season: Season year
        models_dir: Directory containing all trained models
        features_file: Player features JSON (with injury data merged)
        injury_file: Optional injury data JSON
        output_dir: Directory to save reports
        min_r2: Minimum R² threshold for deployment
        min_samples: Minimum samples required for evaluation

    Returns:
        Comprehensive results dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE MULTI-PROP BACKTEST - Season {season}")
    print(f"Models Directory: {models_dir}")
    print(f"Min R² Threshold: {min_r2}")
    print(f"{'='*80}\n")

    # Load player features
    print(f"📂 Loading player features...")
    with open(features_file, 'r') as f:
        player_features = json.load(f)
    print(f"✓ Loaded features for {len(player_features)} players")

    # Load injury data if provided
    injury_map = {}
    if injury_file and injury_file.exists():
        print(f"🏥 Loading injury data...")
        with open(injury_file, 'r') as f:
            injury_map = json.load(f)
        print(f"✓ Loaded {len(injury_map)} injury records")

    # Load all models
    print(f"\n📦 Loading trained models...")
    models = _load_all_models(models_dir)
    print(f"✓ Loaded {len(models)} prop models")

    # Run backtest for each model
    print(f"\n🧪 Running backtests...")
    results = {}
    passed_models = []
    failed_models = []

    for market, model_info in models.items():
        print(f"\n{'─'*80}")
        print(f"Testing: {market}")
        print(f"{'─'*80}")

        try:
            model_results = _backtest_single_model(
                market=market,
                model=model_info['model'],
                player_features=player_features,
                injury_map=injury_map,
                min_samples=min_samples
            )

            results[market] = model_results

            # Check if model passes threshold
            r2 = model_results.get('metrics', {}).get('r2', 0)
            samples = model_results.get('metrics', {}).get('samples', 0)

            if r2 >= min_r2 and samples >= min_samples:
                passed_models.append({
                    'market': market,
                    'r2': r2,
                    'rmse': model_results.get('metrics', {}).get('rmse', 0),
                    'samples': samples
                })
                print(f"✓ PASS: R²={r2:.3f}, RMSE={model_results.get('metrics', {}).get('rmse', 0):.2f}, N={samples}")
            else:
                failed_models.append({
                    'market': market,
                    'r2': r2,
                    'rmse': model_results.get('metrics', {}).get('rmse', 0),
                    'samples': samples,
                    'reason': f'R² below threshold' if r2 < min_r2 else f'Insufficient samples'
                })
                print(f"✗ FAIL: R²={r2:.3f}, RMSE={model_results.get('metrics', {}).get('rmse', 0):.2f}, N={samples}")

        except Exception as e:
            print(f"✗ Error backtesting {market}: {e}")
            failed_models.append({
                'market': market,
                'error': str(e)
            })
            results[market] = {'error': str(e)}

    # Rank models by performance
    ranked_models = sorted(
        passed_models,
        key=lambda x: x['r2'],
        reverse=True
    )

    # Create comprehensive report
    report = {
        'season': season,
        'timestamp': datetime.now().isoformat(),
        'total_models': len(models),
        'passed_models': len(passed_models),
        'failed_models': len(failed_models),
        'pass_rate': len(passed_models) / len(models) if models else 0,

        # Model rankings
        'top_models': ranked_models[:10],
        'worst_models': sorted(passed_models, key=lambda x: x['r2'])[:10],

        # Failed models
        'failed_models_list': failed_models,

        # Category breakdown
        'category_performance': _categorize_results(results),

        # All results
        'all_results': results,

        # Deployment recommendations
        'deployment_ready': [m['market'] for m in passed_models],
        'needs_improvement': [m['market'] for m in failed_models],

        # Summary stats
        'summary': {
            'avg_r2': np.mean([m['r2'] for m in passed_models]) if passed_models else 0,
            'avg_rmse': np.mean([m['rmse'] for m in passed_models]) if passed_models else 0,
            'total_predictions': sum(r.get('metrics', {}).get('samples', 0) for r in results.values()),
        }
    }

    # Save reports
    json_report = output_dir / f'comprehensive_backtest_{season}.json'
    with open(json_report, 'w') as f:
        json.dump(report, f, indent=2)

    summary_report = output_dir / f'backtest_summary_{season}.txt'
    _generate_text_report(report, summary_report)

    print(f"\n✓ Comprehensive backtest complete")
    print(f"✓ JSON report: {json_report}")
    print(f"✓ Summary report: {summary_report}")

    _print_summary(report)

    return report


def _load_all_models(models_dir: Path) -> Dict:
    """Load all trained models.

    Args:
        models_dir: Directory containing model pickle files

    Returns:
        Dict mapping market -> {model, metadata}
    """
    models = {}

    if not models_dir.exists():
        print(f"⚠️  Models directory not found: {models_dir}")
        return models

    for model_file in models_dir.glob('*_model_*.pkl'):
        # Extract market name
        market = model_file.stem.replace('_model_xgboost', '').replace('_model_lightgbm', '')

        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)

            models[market] = {
                'model': model,
                'path': str(model_file),
                'type': 'xgboost' if 'xgboost' in model_file.name else 'lightgbm'
            }

        except Exception as e:
            print(f"⚠️  Failed to load {model_file}: {e}")

    return models


def _backtest_single_model(
    market: str,
    model,
    player_features: Dict,
    injury_map: Dict,
    min_samples: int
) -> Dict:
    """Backtest a single prop model.

    Args:
        market: Market name
        model: Trained model
        player_features: Player feature data
        injury_map: Injury data
        min_samples: Minimum samples required

    Returns:
        Backtest results dict
    """
    # Get target field and features for this market
    # (Simplified - in production, load from PROP_MODEL_CONFIG)
    target_field, feature_cols = _get_model_config(market)

    predictions = []
    actuals = []
    dnp_count = 0
    total_count = 0

    for player_id, games in player_features.items():
        for game in games:
            total_count += 1

            # Skip if player was DNP
            if not game.get('is_active', True):
                dnp_count += 1
                continue

            # Extract features
            features = [game.get(f, 0) for f in feature_cols]

            if not any(features):
                continue

            # Get actual value
            actual = game.get(target_field, 0)

            # Make prediction
            try:
                pred = model.predict([features])[0]
                predictions.append(pred)
                actuals.append(actual)
            except:
                continue

    # Calculate metrics
    if len(predictions) >= min_samples:
        metrics = _calculate_metrics(predictions, actuals)
    else:
        metrics = {
            'error': f'Insufficient samples: {len(predictions)} < {min_samples}',
            'samples': len(predictions)
        }

    return {
        'market': market,
        'metrics': metrics,
        'total_games': total_count,
        'dnp_games': dnp_count,
        'active_games': len(predictions)
    }


def _get_model_config(market: str) -> tuple:
    """Get target field and feature columns for a market.

    Args:
        market: Market name

    Returns:
        Tuple of (target_field, feature_cols)
    """
    # Simplified mapping - in production, load from PROP_MODEL_CONFIG
    if 'pass_yds' in market:
        return 'passing_yards', ['qb_epa', 'cpoe_avg', 'success_rate', 'attempts', 'completions']
    elif 'pass_tds' in market:
        return 'passing_tds', ['qb_epa', 'success_rate', 'attempts']
    elif 'rush_yds' in market:
        return 'rushing_yards', ['rushing_epa', 'rushing_attempts', 'success_rate']
    elif 'rush_tds' in market:
        return 'rushing_tds', ['rushing_epa', 'rushing_attempts']
    elif 'receptions' in market:
        return 'receptions', ['targets', 'receiving_epa']
    elif 'reception_yds' in market:
        return 'receiving_yards', ['receptions', 'targets', 'receiving_epa']
    elif 'reception_tds' in market:
        return 'receiving_tds', ['receptions', 'targets', 'receiving_epa']
    else:
        return 'passing_yards', ['qb_epa', 'attempts']  # Default


def _calculate_metrics(predictions: List[float], actuals: List[float]) -> Dict:
    """Calculate backtest metrics.

    Args:
        predictions: Predicted values
        actuals: Actual values

    Returns:
        Metrics dict
    """
    if not predictions or not actuals:
        return {'error': 'No predictions', 'samples': 0}

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # RMSE
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # MAE
    mae = np.mean(np.abs(predictions - actuals))

    # R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Mean error (bias)
    mean_error = np.mean(predictions - actuals)

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mean_error': float(mean_error),
        'samples': len(predictions)
    }


def _categorize_results(results: Dict) -> Dict:
    """Categorize backtest results by prop type.

    Args:
        results: All backtest results

    Returns:
        Category breakdown
    """
    categories = {
        'passing': [],
        'rushing': [],
        'receiving': [],
        'kicking': [],
        'defense': [],
        'touchdown': [],
        'combo': [],
        'quarter_half': []
    }

    for market, result in results.items():
        if 'error' in result:
            continue

        metrics = result.get('metrics', {})
        if 'error' in metrics:
            continue

        r2 = metrics.get('r2', 0)

        if any(x in market for x in ['1h', '1q', '2h', '3q', '4q']):
            categories['quarter_half'].append({'market': market, 'r2': r2})
        elif 'pass' in market:
            categories['passing'].append({'market': market, 'r2': r2})
        elif 'rush' in market:
            categories['rushing'].append({'market': market, 'r2': r2})
        elif 'reception' in market or 'receptions' in market:
            categories['receiving'].append({'market': market, 'r2': r2})
        elif 'kick' in market or 'field_goal' in market:
            categories['kicking'].append({'market': market, 'r2': r2})
        elif 'tackle' in market or 'sack' in market or 'interception' in market:
            categories['defense'].append({'market': market, 'r2': r2})
        elif 'td' in market:
            categories['touchdown'].append({'market': market, 'r2': r2})
        else:
            categories['combo'].append({'market': market, 'r2': r2})

    # Calculate averages
    summary = {}
    for category, models in categories.items():
        if models:
            summary[category] = {
                'count': len(models),
                'avg_r2': np.mean([m['r2'] for m in models]),
                'best': max(models, key=lambda x: x['r2']),
                'worst': min(models, key=lambda x: x['r2'])
            }

    return summary


def _generate_text_report(report: Dict, output_file: Path):
    """Generate human-readable text report.

    Args:
        report: Backtest report dict
        output_file: Output file path
    """
    lines = []
    lines.append("=" * 80)
    lines.append("COMPREHENSIVE MULTI-PROP BACKTEST REPORT")
    lines.append("=" * 80)
    lines.append(f"\nSeason: {report['season']}")
    lines.append(f"Timestamp: {report['timestamp']}")
    lines.append(f"\nTotal Models: {report['total_models']}")
    lines.append(f"Passed: {report['passed_models']} ({report['pass_rate']*100:.1f}%)")
    lines.append(f"Failed: {report['failed_models']}")

    lines.append(f"\n{'='*80}")
    lines.append("TOP 10 MODELS")
    lines.append("=" * 80)
    for i, model in enumerate(report['top_models'], 1):
        lines.append(f"{i:2d}. {model['market']:40s} R²={model['r2']:.3f}, RMSE={model['rmse']:.2f}")

    lines.append(f"\n{'='*80}")
    lines.append("CATEGORY PERFORMANCE")
    lines.append("=" * 80)
    for category, stats in report['category_performance'].items():
        lines.append(f"\n{category.upper()}:")
        lines.append(f"  Models: {stats['count']}")
        lines.append(f"  Avg R²: {stats['avg_r2']:.3f}")
        lines.append(f"  Best: {stats['best']['market']} (R²={stats['best']['r2']:.3f})")
        lines.append(f"  Worst: {stats['worst']['market']} (R²={stats['worst']['r2']:.3f})")

    if report['failed_models_list']:
        lines.append(f"\n{'='*80}")
        lines.append("FAILED MODELS")
        lines.append("=" * 80)
        for model in report['failed_models_list']:
            reason = model.get('reason', model.get('error', 'Unknown'))
            lines.append(f"- {model['market']}: {reason}")

    lines.append(f"\n{'='*80}")
    lines.append("DEPLOYMENT RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append(f"\nREADY FOR DEPLOYMENT ({len(report['deployment_ready'])} models):")
    for market in report['deployment_ready'][:20]:
        lines.append(f"  ✓ {market}")

    lines.append(f"\nNEEDS IMPROVEMENT ({len(report['needs_improvement'])} models):")
    for market in report['needs_improvement'][:20]:
        lines.append(f"  ✗ {market}")

    lines.append(f"\n{'='*80}\n")

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))


def _print_summary(report: Dict):
    """Print summary to console.

    Args:
        report: Backtest report
    """
    print(f"\n{'='*80}")
    print("BACKTEST SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal Models: {report['total_models']}")
    print(f"Passed: {report['passed_models']} ({report['pass_rate']*100:.1f}%)")
    print(f"Failed: {report['failed_models']}")
    print(f"\nAvg R²: {report['summary']['avg_r2']:.3f}")
    print(f"Avg RMSE: {report['summary']['avg_rmse']:.2f}")
    print(f"Total Predictions: {report['summary']['total_predictions']}")

    print(f"\nTop 5 Models:")
    for i, model in enumerate(report['top_models'][:5], 1):
        print(f"  {i}. {model['market']:40s} R²={model['r2']:.3f}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run comprehensive backtest on all multi-prop models'
    )
    parser.add_argument('--season', type=int, required=True,
                       help='Season year')
    parser.add_argument('--models-dir', type=Path,
                       default=Path('outputs/models/multi_prop'),
                       help='Directory containing trained models')
    parser.add_argument('--features-file', type=Path, required=True,
                       help='Player features JSON (with injury data merged)')
    parser.add_argument('--injury-file', type=Path,
                       help='Optional injury data JSON')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('outputs/backtest/comprehensive'),
                       help='Output directory for reports')
    parser.add_argument('--min-r2', type=float, default=0.3,
                       help='Minimum R² threshold (default: 0.3)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum samples per model (default: 10)')

    args = parser.parse_args()

    run_comprehensive_backtest(
        season=args.season,
        models_dir=args.models_dir,
        features_file=args.features_file,
        injury_file=args.injury_file,
        output_dir=args.output_dir,
        min_r2=args.min_r2,
        min_samples=args.min_samples
    )
