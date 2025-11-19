"""Prop value detection system - Find mispriced props using model projections vs market odds.

This module identifies +EV (positive expected value) betting opportunities by:
1. Loading all trained multi-prop models (60+ markets)
2. Generating projections for upcoming games
3. Fetching current DraftKings odds from prop line snapshots
4. Comparing model projections vs market lines
5. Calculating expected value and win probability
6. Filtering for injury status (exclude Out/Doubtful players)
7. Ranking props by edge size and confidence

Expected Value Calculation:
- EV = (Win_Prob * Over_Payout) + (Lose_Prob * Loss)
- Positive EV = Model projection significantly different from market line
- Edge = Model's implied win% - Market's implied win%

Confidence Levels:
- HIGH: Edge > 10%, Model R² > 0.6, Active player
- MEDIUM: Edge > 5%, Model R² > 0.4, Expected to play
- LOW: Edge > 2%, Model R² > 0.3, Questionable status
"""

from pathlib import Path
import argparse
import json
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
from datetime import datetime


def detect_prop_value(
    models_dir: Path,
    current_odds_file: Path,
    upcoming_features_file: Path,
    injury_file: Optional[Path],
    output_file: Path,
    min_edge: float = 0.02,  # Minimum 2% edge
    confidence_level: str = 'medium'
) -> Dict:
    """Detect mispriced props using model projections vs market odds.

    Args:
        models_dir: Directory containing trained models
        current_odds_file: JSON file with current DraftKings odds
        upcoming_features_file: Player features for upcoming games
        injury_file: Optional injury data
        output_file: Path to save value detection report
        min_edge: Minimum edge threshold (default 2%)
        confidence_level: Filter level (low/medium/high)

    Returns:
        Dict with value opportunities
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PROP VALUE DETECTION")
    print(f"Models: {models_dir}")
    print(f"Min Edge: {min_edge*100:.1f}%")
    print(f"Confidence: {confidence_level.upper()}")
    print(f"{'='*80}\n")

    # Load all trained models
    print(f"📦 Loading trained models...")
    models = _load_all_models(models_dir)
    print(f"✓ Loaded {len(models)} prop models")

    # Load current odds
    print(f"📊 Loading current odds...")
    with open(current_odds_file, 'r') as f:
        current_odds = json.load(f)
    print(f"✓ Loaded odds for {len(current_odds.get('games', []))} games")

    # Load upcoming game features
    print(f"📂 Loading player features...")
    with open(upcoming_features_file, 'r') as f:
        player_features = json.load(f)
    print(f"✓ Loaded features for {len(player_features)} players")

    # Load injury data if provided
    injury_map = {}
    if injury_file and injury_file.exists():
        print(f"🏥 Loading injury data...")
        with open(injury_file, 'r') as f:
            injury_map = json.load(f)
        print(f"✓ Loaded {len(injury_map)} injury records")

    # Generate projections
    print(f"\n🔮 Generating model projections...")
    projections = _generate_all_projections(models, player_features)
    print(f"✓ Generated {len(projections)} projections")

    # Compare projections vs odds
    print(f"\n💰 Comparing projections vs market odds...")
    value_opportunities = _find_value_opportunities(
        projections=projections,
        current_odds=current_odds,
        injury_map=injury_map,
        min_edge=min_edge,
        confidence_level=confidence_level
    )

    # Rank by edge
    value_opportunities_ranked = sorted(
        value_opportunities,
        key=lambda x: abs(x['edge']),
        reverse=True
    )

    # Create report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_projections': len(projections),
        'total_opportunities': len(value_opportunities_ranked),
        'min_edge': min_edge,
        'confidence_level': confidence_level,

        # Top opportunities
        'top_over_plays': [
            opp for opp in value_opportunities_ranked
            if opp['recommendation'] == 'OVER'
        ][:20],

        'top_under_plays': [
            opp for opp in value_opportunities_ranked
            if opp['recommendation'] == 'UNDER'
        ][:20],

        # All opportunities (for further analysis)
        'all_opportunities': value_opportunities_ranked,

        # Summary stats
        'summary': {
            'over_opportunities': len([o for o in value_opportunities_ranked if o['recommendation'] == 'OVER']),
            'under_opportunities': len([o for o in value_opportunities_ranked if o['recommendation'] == 'UNDER']),
            'high_confidence': len([o for o in value_opportunities_ranked if o['confidence'] == 'HIGH']),
            'medium_confidence': len([o for o in value_opportunities_ranked if o['confidence'] == 'MEDIUM']),
            'low_confidence': len([o for o in value_opportunities_ranked if o['confidence'] == 'LOW']),
            'avg_edge': np.mean([abs(o['edge']) for o in value_opportunities_ranked]) if value_opportunities_ranked else 0
        }
    }

    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Value detection complete: {output_file}")
    _print_summary(report)

    return report


def _load_all_models(models_dir: Path) -> Dict:
    """Load all trained prop models.

    Args:
        models_dir: Directory containing model pickle files

    Returns:
        Dict mapping market -> model
    """
    models = {}

    if not models_dir.exists():
        print(f"⚠️  Models directory not found: {models_dir}")
        return models

    for model_file in models_dir.glob('*_model_*.pkl'):
        # Extract market name from filename
        # Format: player_pass_yds_model_xgboost.pkl
        market = model_file.stem.replace('_model_xgboost', '').replace('_model_lightgbm', '')

        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                models[market] = model
        except Exception as e:
            print(f"⚠️  Failed to load {model_file}: {e}")

    return models


def _generate_all_projections(
    models: Dict,
    player_features: Dict
) -> List[Dict]:
    """Generate projections for all players and markets.

    Args:
        models: Dict of trained models
        player_features: Player feature data

    Returns:
        List of projection dicts
    """
    projections = []

    for market, model in models.items():
        # Get feature columns needed for this model (simplified)
        # In production, would load from model metadata
        feature_cols = _get_model_features(market)

        for player_id, games in player_features.items():
            if not games:
                continue

            # Use latest game features for projection
            latest_game = games[-1]

            # Extract features
            features = [latest_game.get(f, 0) for f in feature_cols]

            if not any(features):
                continue

            # Make prediction
            try:
                projection = model.predict([features])[0]

                projections.append({
                    'player_id': player_id,
                    'player_name': latest_game.get('player_name', player_id),
                    'position': latest_game.get('position', 'Unknown'),
                    'team': latest_game.get('team', 'Unknown'),
                    'market': market,
                    'projection': round(projection, 2),
                    'week': latest_game.get('week', 0),
                    'season': latest_game.get('season', 0)
                })

            except Exception as e:
                # Skip if prediction fails
                continue

    return projections


def _find_value_opportunities(
    projections: List[Dict],
    current_odds: Dict,
    injury_map: Dict,
    min_edge: float,
    confidence_level: str
) -> List[Dict]:
    """Find value opportunities by comparing projections vs odds.

    Args:
        projections: Model projections
        current_odds: Current DraftKings odds
        injury_map: Injury data
        min_edge: Minimum edge threshold
        confidence_level: Filter level

    Returns:
        List of value opportunity dicts
    """
    opportunities = []

    # Build quick lookup for odds by player/market
    odds_lookup = {}
    for game in current_odds.get('games', []):
        for market, props in game.get('props_by_market', {}).items():
            for prop in props:
                key = f"{prop['player']}_{market}"
                odds_lookup[key] = {
                    'line': prop['line'],
                    'over_odds': prop['over_odds'],
                    'under_odds': prop['under_odds'],
                    'game_id': game['game_id']
                }

    # Compare each projection to market odds
    for proj in projections:
        player_name = proj['player_name']
        market = proj['market']
        projection = proj['projection']

        # Look up market odds
        odds_key = f"{player_name}_{market}"
        if odds_key not in odds_lookup:
            continue

        odds_info = odds_lookup[odds_key]
        market_line = odds_info['line']
        over_odds = odds_info['over_odds']
        under_odds = odds_info['under_odds']

        # Check injury status
        season = proj['season']
        week = proj['week']
        player_id = proj['player_id']
        injury_key = f"{season}_{week}_{player_id}"
        injury_info = injury_map.get(injury_key, {})
        injury_status = injury_info.get('report_status', 'Healthy')
        expected_to_play = injury_info.get('expected_to_play', True)

        # Skip if player is Out or Doubtful
        if injury_status in ['Out', 'Doubtful']:
            continue

        # Calculate edge
        diff = projection - market_line
        edge_pct = diff / market_line if market_line > 0 else 0

        # Determine recommendation
        if abs(edge_pct) < min_edge:
            continue

        if edge_pct > 0:
            # Model projects higher than market → OVER
            recommendation = 'OVER'
            odds_to_use = over_odds
            edge = edge_pct
        else:
            # Model projects lower than market → UNDER
            recommendation = 'UNDER'
            odds_to_use = under_odds
            edge = abs(edge_pct)

        # Calculate expected value
        implied_prob = _american_odds_to_prob(odds_to_use)
        model_prob = _calculate_model_win_prob(projection, market_line, recommendation)
        ev = _calculate_ev(model_prob, odds_to_use)

        # Determine confidence
        confidence = _determine_confidence(
            edge=edge,
            injury_status=injury_status,
            expected_to_play=expected_to_play,
            model_r2=0.5  # TODO: Load from model metadata
        )

        # Filter by confidence level
        if confidence_level == 'high' and confidence != 'HIGH':
            continue
        elif confidence_level == 'medium' and confidence == 'LOW':
            continue

        # Create opportunity
        opportunity = {
            'player_name': player_name,
            'position': proj['position'],
            'team': proj['team'],
            'market': market,
            'game_id': odds_info['game_id'],

            # Model vs Market
            'projection': round(projection, 2),
            'market_line': round(market_line, 2),
            'difference': round(diff, 2),
            'edge': round(edge, 4),
            'edge_pct': f"{edge*100:.2f}%",

            # Recommendation
            'recommendation': recommendation,
            'odds': odds_to_use,
            'implied_prob': round(implied_prob, 4),
            'model_prob': round(model_prob, 4),
            'expected_value': round(ev, 4),
            'ev_pct': f"{ev*100:.2f}%",

            # Risk factors
            'injury_status': injury_status,
            'expected_to_play': expected_to_play,
            'confidence': confidence,

            # Metadata
            'week': week,
            'season': season
        }

        opportunities.append(opportunity)

    return opportunities


def _get_model_features(market: str) -> List[str]:
    """Get feature columns for a market (simplified).

    Args:
        market: Market name

    Returns:
        List of feature column names
    """
    # Simplified feature mapping
    # In production, load from PROP_MODEL_CONFIG
    if 'pass' in market:
        return ['qb_epa', 'cpoe_avg', 'success_rate', 'attempts', 'completions']
    elif 'rush' in market:
        return ['rushing_epa', 'rushing_attempts', 'success_rate']
    elif 'reception' in market or 'receptions' in market:
        return ['targets', 'receiving_epa', 'receptions']
    else:
        return ['total_epa', 'total_plays']


def _american_odds_to_prob(american_odds: int) -> float:
    """Convert American odds to implied probability.

    Args:
        american_odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability (0-1)
    """
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def _calculate_model_win_prob(projection: float, line: float, side: str) -> float:
    """Calculate model's implied win probability.

    Args:
        projection: Model projection
        line: Market line
        side: 'OVER' or 'UNDER'

    Returns:
        Win probability (0-1)
    """
    # Simplified: Assume normal distribution with std dev = 20% of projection
    std_dev = max(projection * 0.2, 10)  # At least 10 yards std dev

    if side == 'OVER':
        # P(actual > line)
        z_score = (projection - line) / std_dev
    else:
        # P(actual < line)
        z_score = (line - projection) / std_dev

    # Convert z-score to probability (simplified)
    # In production, use scipy.stats.norm.cdf
    prob = 0.5 + 0.5 * np.tanh(z_score / 2)

    return max(0.01, min(0.99, prob))


def _calculate_ev(win_prob: float, american_odds: int) -> float:
    """Calculate expected value.

    Args:
        win_prob: Win probability (0-1)
        american_odds: American odds

    Returns:
        Expected value as decimal (e.g., 0.05 = 5% EV)
    """
    if american_odds < 0:
        payout = 100 / abs(american_odds)
    else:
        payout = american_odds / 100

    ev = (win_prob * payout) - (1 - win_prob)

    return ev


def _determine_confidence(
    edge: float,
    injury_status: str,
    expected_to_play: bool,
    model_r2: float
) -> str:
    """Determine confidence level for opportunity.

    Args:
        edge: Edge size (0-1)
        injury_status: Injury status
        expected_to_play: Whether player expected to play
        model_r2: Model R² score

    Returns:
        Confidence level: 'HIGH', 'MEDIUM', 'LOW'
    """
    if edge > 0.10 and model_r2 > 0.6 and injury_status == 'Healthy':
        return 'HIGH'
    elif edge > 0.05 and model_r2 > 0.4 and expected_to_play:
        return 'MEDIUM'
    else:
        return 'LOW'


def _print_summary(report: Dict):
    """Print formatted summary of value opportunities.

    Args:
        report: Value detection report
    """
    print(f"\n{'='*80}")
    print("VALUE DETECTION SUMMARY")
    print(f"{'='*80}")

    summary = report['summary']
    print(f"\nTotal Opportunities: {report['total_opportunities']}")
    print(f"  OVER plays: {summary['over_opportunities']}")
    print(f"  UNDER plays: {summary['under_opportunities']}")

    print(f"\nConfidence Breakdown:")
    print(f"  HIGH: {summary['high_confidence']}")
    print(f"  MEDIUM: {summary['medium_confidence']}")
    print(f"  LOW: {summary['low_confidence']}")

    print(f"\nAverage Edge: {summary['avg_edge']*100:.2f}%")

    print(f"\nTop 5 OVER Plays:")
    for i, opp in enumerate(report['top_over_plays'][:5], 1):
        print(f"  {i}. {opp['player_name']} {opp['market']}: "
              f"Proj {opp['projection']} vs Line {opp['market_line']} "
              f"(Edge: {opp['edge_pct']}, EV: {opp['ev_pct']}, {opp['confidence']})")

    print(f"\nTop 5 UNDER Plays:")
    for i, opp in enumerate(report['top_under_plays'][:5], 1):
        print(f"  {i}. {opp['player_name']} {opp['market']}: "
              f"Proj {opp['projection']} vs Line {opp['market_line']} "
              f"(Edge: {opp['edge_pct']}, EV: {opp['ev_pct']}, {opp['confidence']})")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detect mispriced props using model projections vs market odds'
    )
    parser.add_argument('--models-dir', type=Path,
                       default=Path('outputs/models/multi_prop'),
                       help='Directory containing trained models')
    parser.add_argument('--current-odds', type=Path, required=True,
                       help='JSON file with current DraftKings odds')
    parser.add_argument('--features-file', type=Path, required=True,
                       help='Player features for upcoming games')
    parser.add_argument('--injury-file', type=Path,
                       help='Optional injury data JSON')
    parser.add_argument('--output', type=Path,
                       default=Path('outputs/analysis/value_opportunities.json'),
                       help='Output path for value detection report')
    parser.add_argument('--min-edge', type=float, default=0.02,
                       help='Minimum edge threshold (default: 0.02 = 2%%)')
    parser.add_argument('--confidence', default='medium',
                       choices=['low', 'medium', 'high'],
                       help='Confidence level filter')

    args = parser.parse_args()

    detect_prop_value(
        models_dir=args.models_dir,
        current_odds_file=args.current_odds,
        upcoming_features_file=args.features_file,
        injury_file=args.injury_file,
        output_file=args.output,
        min_edge=args.min_edge,
        confidence_level=args.confidence
    )
