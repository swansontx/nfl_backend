"""Meta "Trust" Model - Predict When Your Model Is Actually Right

PROBLEM:
  Your model says: "OVER 275.5 yards, 8% edge, HIGH CONFIDENCE"
  But historically: You're only 45% accurate on this prop type + player role combo

  Should you bet? NO! Even though edge looks good.

SOLUTION: Meta Model
  Train a classifier to predict: "Will THIS specific bet actually win?"

  Features:
    - Prop type (pass_yds, rush_yds, etc.)
    - Player role (QB1, WR1, committee RB, etc.)
    - Model edge size (2%, 5%, 10%, etc.)
    - Model confidence (variance of prediction)
    - Injury status (Healthy, Questionable, etc.)
    - Game type (Primetime, divisional, weather, etc.)
    - Historical accuracy for this player + prop combo
    - CLV history (do we beat closing line on this prop type?)

  Target: Did bet win? (binary: 0 or 1)

  Output: "Trust score" (0-1)
    - 0.65+ = High trust → BET
    - 0.50-0.65 = Medium trust → CONSIDER
    - <0.50 = Low trust → SKIP (even if model shows edge)

RESEARCH:
  - Sharp bettors use meta-models to filter noisy prop types
  - Reduces bet volume by 30-40% but increases ROI by 20-30%
  - Key insight: "Not all edges are created equal"

EXAMPLE:
  Base model: "CMC OVER 95.5 rush yards, 6% edge"
  Meta model features:
    - prop_type: rush_yds (historically 58% accurate)
    - player_role: bellcow_rb (stable)
    - edge_size: 0.06 (moderate)
    - injury_status: healthy
    - primetime_game: 1 (we're good on primetime)
  → Trust score: 0.72 (HIGH) → BET

  Base model: "WR3 OVER 3.5 receptions, 8% edge"
  Meta model features:
    - prop_type: receptions (historically 48% accurate)
    - player_role: wr3 (volatile)
    - edge_size: 0.08 (looks good)
    - injury_status: questionable
    - targets_variance: high (inconsistent)
  → Trust score: 0.42 (LOW) → SKIP despite edge

This is the FINAL FILTER before placing bets.
"""

from pathlib import Path
import argparse
import json
import pickle
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve


def train_meta_trust_model(
    bet_history_file: Path,
    output_dir: Path,
    model_type: str = 'random_forest'
) -> Dict:
    """Train meta model to predict bet success.

    Args:
        bet_history_file: JSON file with historical bet results
        output_dir: Output directory for model
        model_type: 'random_forest' or 'logistic'

    Returns:
        Training results dict
    """
    print(f"\n{'='*80}")
    print("TRAINING META TRUST MODEL")
    print(f"Model Type: {model_type}")
    print(f"{'='*80}\n")

    # Load bet history
    with open(bet_history_file, 'r') as f:
        bet_history = json.load(f)

    if not bet_history:
        return {'error': 'No bet history available'}

    # Filter to resulted bets only
    resulted_bets = [b for b in bet_history if b.get('status') == 'resulted' and b.get('won') is not None]

    if len(resulted_bets) < 50:
        return {'error': f'Insufficient bet history: {len(resulted_bets)} bets'}

    print(f"✓ Loaded {len(resulted_bets)} resulted bets")

    # Prepare training data
    df = _prepare_meta_training_data(resulted_bets)

    print(f"✓ Prepared {len(df)} samples")

    # Define features
    feature_cols = [
        # Prop characteristics
        'prop_type_pass', 'prop_type_rush', 'prop_type_rec', 'prop_type_td',

        # Player role (TODO: infer from bet history)
        'is_qb', 'is_rb', 'is_wr',

        # Bet characteristics
        'edge_size', 'edge_bucket_low', 'edge_bucket_medium', 'edge_bucket_high',
        'side_over', 'side_under',

        # Model confidence (if available)
        'model_projection', 'opening_line', 'projection_line_diff',

        # Market signals
        'clv', 'clv_positive', 'opening_odds',

        # Game context (if available)
        'spread', 'total', 'is_favorite',
        'is_dome', 'wind_high', 'temp_cold',
        'is_primetime',

        # Historical performance (rolling)
        'recent_win_rate', 'prop_type_win_rate'
    ]

    # Filter to available features
    feature_cols_available = [f for f in feature_cols if f in df.columns]

    X = df[feature_cols_available].fillna(0)
    y = df['won'].astype(int)

    print(f"✓ Features: {len(feature_cols_available)}")
    print(f"✓ Win rate: {y.mean()*100:.1f}%")

    # Time-based split (important: no shuffling!)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}")

    # Train meta model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
        )
    else:  # logistic
        model = LogisticRegression(
            C=1.0,
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )

    model.fit(X_train, y_train)

    # Evaluate
    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, train_probs)
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"\n✓ Train AUC: {train_auc:.3f}")
    print(f"✓ Val AUC: {val_auc:.3f}")

    # Calibration analysis
    calibration = _analyze_calibration(y_val, val_probs)

    print(f"\nCalibration by Trust Score:")
    for bucket, metrics in calibration.items():
        print(f"  {bucket}: Win Rate = {metrics['win_rate']*100:.1f}%, N={metrics['count']}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = sorted(
            zip(feature_cols_available, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"\nTop 10 Features:")
        for feat, importance in feature_importance[:10]:
            print(f"  {feat:30s}: {importance:.4f}")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f'meta_trust_model_{model_type}.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': feature_cols_available,
            'model_type': model_type,
            'calibration': calibration,
            'feature_importance': feature_importance if hasattr(model, 'feature_importances_') else None
        }, f)

    print(f"\n✓ Saved meta trust model to: {model_path}")

    return {
        'model_path': str(model_path),
        'train_auc': train_auc,
        'val_auc': val_auc,
        'calibration': calibration,
        'features_used': feature_cols_available,
        'samples': len(df)
    }


def predict_trust_score(
    meta_model_path: Path,
    bet_features: Dict
) -> Dict:
    """Predict trust score for a potential bet.

    Args:
        meta_model_path: Path to meta trust model
        bet_features: Features of the bet

    Returns:
        Trust score and recommendation
    """
    # Load meta model
    with open(meta_model_path, 'rb') as f:
        meta_bundle = pickle.load(f)

    model = meta_bundle['model']
    feature_names = meta_bundle['features']

    # Extract features
    feature_values = [bet_features.get(f, 0) for f in feature_names]
    X = np.array(feature_values).reshape(1, -1)

    # Predict probability
    trust_score = model.predict_proba(X)[0][1]

    # Determine recommendation
    if trust_score >= 0.65:
        recommendation = 'BET'
        confidence_level = 'HIGH'
    elif trust_score >= 0.50:
        recommendation = 'CONSIDER'
        confidence_level = 'MEDIUM'
    else:
        recommendation = 'SKIP'
        confidence_level = 'LOW'

    return {
        'trust_score': round(trust_score, 3),
        'recommendation': recommendation,
        'confidence_level': confidence_level
    }


def _prepare_meta_training_data(bet_history: List[Dict]) -> pd.DataFrame:
    """Prepare training data for meta model.

    Args:
        bet_history: List of resulted bets

    Returns:
        DataFrame with features and target
    """
    rows = []

    # Calculate rolling win rates
    prop_type_wins = {}
    prop_type_counts = {}
    rolling_wins = []
    rolling_counts = []

    for bet in bet_history:
        prop_type = bet.get('prop_type', '')
        won = bet.get('won', False)

        # Update rolling stats
        rolling_wins.append(won)
        rolling_counts.append(1)

        # Update prop type stats
        if prop_type not in prop_type_wins:
            prop_type_wins[prop_type] = 0
            prop_type_counts[prop_type] = 0

        prop_type_wins[prop_type] += int(won)
        prop_type_counts[prop_type] += 1

        # Calculate rolling win rate (last 20 bets)
        recent_win_rate = sum(rolling_wins[-20:]) / len(rolling_wins[-20:]) if rolling_wins else 0.5

        # Calculate prop type win rate
        prop_type_win_rate = (
            prop_type_wins[prop_type] / prop_type_counts[prop_type]
            if prop_type_counts[prop_type] > 0 else 0.5
        )

        # Extract features
        row = {
            'bet_id': bet.get('bet_id', ''),
            'won': int(won),

            # Prop type one-hot
            'prop_type_pass': 1 if 'pass' in prop_type else 0,
            'prop_type_rush': 1 if 'rush' in prop_type else 0,
            'prop_type_rec': 1 if 'rec' in prop_type else 0,
            'prop_type_td': 1 if 'td' in prop_type else 0,

            # Player role (simplified inference)
            'is_qb': 1 if 'pass' in prop_type else 0,
            'is_rb': 1 if 'rush' in prop_type else 0,
            'is_wr': 1 if 'reception' in prop_type else 0,

            # Bet characteristics
            'edge_size': bet.get('model_edge', 0),
            'edge_bucket_low': 1 if bet.get('model_edge', 0) < 0.04 else 0,
            'edge_bucket_medium': 1 if 0.04 <= bet.get('model_edge', 0) < 0.08 else 0,
            'edge_bucket_high': 1 if bet.get('model_edge', 0) >= 0.08 else 0,

            'side_over': 1 if bet.get('side', '') == 'over' else 0,
            'side_under': 1 if bet.get('side', '') == 'under' else 0,

            # Model confidence
            'model_projection': bet.get('model_projection', 0),
            'opening_line': bet.get('opening_line', 0),
            'projection_line_diff': bet.get('model_projection', 0) - bet.get('opening_line', 0),

            # Market signals
            'clv': bet.get('clv', 0),
            'clv_positive': 1 if bet.get('clv', 0) > 0 else 0,
            'opening_odds': bet.get('opening_odds', -110),

            # Game context (if available in bet record)
            'spread': 0,  # TODO: add if available
            'total': 0,
            'is_favorite': 0,
            'is_dome': 0,
            'wind_high': 0,
            'temp_cold': 0,
            'is_primetime': 0,

            # Historical performance
            'recent_win_rate': recent_win_rate,
            'prop_type_win_rate': prop_type_win_rate
        }

        rows.append(row)

    return pd.DataFrame(rows)


def _analyze_calibration(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
    """Analyze calibration of trust scores.

    Args:
        y_true: True outcomes (0 or 1)
        y_pred_proba: Predicted probabilities

    Returns:
        Calibration dict by score buckets
    """
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred_proba
    })

    # Bucket by predicted probability
    df['bucket'] = pd.cut(
        df['predicted'],
        bins=[0, 0.45, 0.55, 0.65, 1.0],
        labels=['Low (<0.45)', 'Medium (0.45-0.55)', 'High (0.55-0.65)', 'Very High (0.65+)']
    )

    calibration = {}

    for bucket in df['bucket'].unique():
        bucket_df = df[df['bucket'] == bucket]

        if len(bucket_df) > 0:
            calibration[str(bucket)] = {
                'count': len(bucket_df),
                'win_rate': bucket_df['actual'].mean(),
                'avg_predicted': bucket_df['predicted'].mean()
            }

    return calibration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train meta trust model to predict bet success'
    )
    parser.add_argument('--action', required=True,
                       choices=['train', 'predict'],
                       help='Action to perform')
    parser.add_argument('--bet-history', type=Path,
                       default=Path('outputs/betting/clv_bets.json'),
                       help='Bet history JSON (from CLV tracker)')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('outputs/models/meta_trust'),
                       help='Output directory for meta model')
    parser.add_argument('--model-type', default='random_forest',
                       choices=['random_forest', 'logistic'],
                       help='Meta model type')
    parser.add_argument('--meta-model', type=Path,
                       help='Path to trained meta model (for predict)')
    parser.add_argument('--bet-features-json', type=Path,
                       help='JSON with bet features (for predict)')

    args = parser.parse_args()

    if args.action == 'train':
        # Train meta trust model
        result = train_meta_trust_model(
            bet_history_file=args.bet_history,
            output_dir=args.output_dir,
            model_type=args.model_type
        )

        print(f"\n{'='*80}")
        print("META TRUST MODEL TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"\nVal AUC: {result['val_auc']:.3f}")
        print(f"Model: {result['model_path']}")

    elif args.action == 'predict':
        # Predict trust score for a bet
        with open(args.bet_features_json, 'r') as f:
            bet_features = json.load(f)

        trust_result = predict_trust_score(
            meta_model_path=args.meta_model,
            bet_features=bet_features
        )

        print(f"\nTrust Score: {trust_result['trust_score']:.3f}")
        print(f"Recommendation: {trust_result['recommendation']}")
        print(f"Confidence: {trust_result['confidence_level']}")
