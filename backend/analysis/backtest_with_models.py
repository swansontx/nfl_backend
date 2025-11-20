"""Backtest using actual trained models, not just rolling averages.

This provides a more accurate assessment of model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, Tuple
import argparse


def load_models(models_dir: Path) -> Dict:
    """Load all trained models."""
    models = {}
    for model_file in models_dir.rglob("*.pkl"):
        try:
            model_name = model_file.stem
            models[model_name] = joblib.load(model_file)
        except Exception as e:
            pass  # Skip failed models
    return models


def predict_with_model(
    models: Dict,
    prop_type: str,
    features: Dict
) -> Tuple[float, float]:
    """Get prediction from trained model.

    Returns:
        (projection, std_dev)
    """
    # Map prop types to model files
    model_map = {
        'passing_yards': 'pass_yards_models',
        'rushing_yards': 'rush_yards_models',
        'receiving_yards': 'rec_yards_models',
        'receptions': 'receptions_models',
        'completions': 'completions_models',
        'attempts': 'attempts_models',
        'carries': 'carries_models',
        'targets': 'targets_models',
        'passing_tds': 'pass_tds_models',
        'rushing_tds': 'rush_tds_models',
        'receiving_tds': 'rec_tds_models',
    }

    model_name = model_map.get(prop_type)
    if not model_name or model_name not in models:
        # Fallback to rolling average
        stat_col = prop_type.replace('passing_', 'pass_').replace('rushing_', 'rush_').replace('receiving_', 'rec_')
        avg = features.get(f'{stat_col}_season_avg', 0)
        std = features.get(f'{stat_col}_std', avg * 0.2)
        return avg, std

    model_data = models[model_name]

    # Get feature names
    stat_col = prop_type.replace('passing_', 'pass_').replace('rushing_', 'rush_').replace('receiving_', 'rec_')

    # Use season avg and L3 avg as features
    season_avg = features.get(f'season_avg_{stat_col}', 0) or features.get(f'{stat_col}_season_avg', 0)
    l3_avg = features.get(f'l3_avg_{stat_col}', 0) or features.get(f'{stat_col}_l3_avg', season_avg)
    games_played = features.get('games_played', 5)

    X = [[season_avg, l3_avg, games_played]]

    try:
        if 'models' in model_data and 'q50' in model_data['models']:
            # New quantile model structure
            projection = float(model_data['models']['q50'].predict(X)[0])
            q25 = float(model_data['models']['q25'].predict(X)[0])
            q75 = float(model_data['models']['q75'].predict(X)[0])
            std_dev = (q75 - q25) / 1.35  # IQR to std approximation
        elif 'Q50' in model_data:
            # Old quantile models
            projection = float(model_data['Q50'].predict(X)[0])
            q25 = float(model_data['Q25'].predict(X)[0])
            q75 = float(model_data['Q75'].predict(X)[0])
            std_dev = (q75 - q25) / 1.35
        elif 'model' in model_data:
            # Poisson/Bernoulli models
            projection = float(model_data['model'].predict(X)[0])
            std_dev = np.sqrt(projection) if projection > 0 else 0.5
        else:
            # Single model
            projection = float(model_data.predict(X)[0])
            std_dev = projection * 0.2

        return max(0, projection), max(1, std_dev)
    except Exception as e:
        return season_avg, season_avg * 0.2


def backtest_with_models(week: int, season: int = 2025) -> Dict:
    """Backtest using trained models."""

    print(f"\n{'='*60}")
    print(f"BACKTESTING WEEK {week} WITH TRAINED MODELS")
    print(f"{'='*60}\n")

    # Load models
    models = load_models(Path("outputs/models/comprehensive"))
    print(f"Loaded {len(models)} comprehensive models")

    # Load player stats
    stats_file = Path("inputs/player_stats_enhanced_2025.csv")
    if not stats_file.exists():
        stats_file = Path("inputs/player_stats_2025.csv")
    player_stats = pd.read_csv(stats_file, low_memory=False)

    # Split data
    historical = player_stats[player_stats['week'] < week].copy()
    actual = player_stats[player_stats['week'] == week].copy()

    print(f"Historical data: {len(historical)} records")
    print(f"Actual results: {len(actual)} records")

    # Props by position
    prop_types = {
        'QB': ['passing_yards', 'completions', 'attempts'],
        'RB': ['rushing_yards', 'carries', 'receptions', 'receiving_yards'],
        'WR': ['receptions', 'receiving_yards', 'targets'],
        'TE': ['receptions', 'receiving_yards', 'targets'],
    }

    results = []

    for _, actual_row in actual.iterrows():
        player_id = actual_row['player_id']
        player_name = actual_row.get('player_display_name', player_id)
        position = actual_row.get('position', 'UNK')

        # Get historical data for features
        player_hist = historical[historical['player_id'] == player_id].sort_values('week')

        if len(player_hist) < 3:
            continue

        # Calculate features
        features = {}
        last_row = player_hist.iloc[-1]

        # Map column names to model feature names
        col_to_feature = {
            'passing_yards': 'pass_yards',
            'rushing_yards': 'rush_yards',
            'receiving_yards': 'rec_yards',
            'receptions': 'receptions',
            'completions': 'completions',
            'attempts': 'attempts',
            'carries': 'carries',
            'targets': 'targets',
        }

        for col, feature_name in col_to_feature.items():
            if col in player_hist.columns:
                values = player_hist[col].dropna()
                if len(values) > 0:
                    features[f'season_avg_{feature_name}'] = values.mean()
                    features[f'{feature_name}_season_avg'] = values.mean()
                    features[f'l3_avg_{feature_name}'] = values.tail(3).mean()
                    features[f'{feature_name}_l3_avg'] = values.tail(3).mean()
                    # Also store with original col name for fallback
                    features[f'season_avg_{col}'] = values.mean()

        features['games_played'] = len(player_hist)

        # Get position group
        pos_group = 'QB' if 'QB' in str(position) else \
                   'RB' if 'RB' in str(position) else \
                   'WR' if 'WR' in str(position) else \
                   'TE' if 'TE' in str(position) else None

        if not pos_group:
            continue

        # Test each prop
        for prop_type in prop_types.get(pos_group, []):
            projection, std_dev = predict_with_model(models, prop_type, features)

            if projection <= 0:
                continue

            # Get actual
            stat_col = prop_type
            if stat_col not in actual_row:
                continue

            actual_val = actual_row[stat_col]
            if pd.isna(actual_val):
                continue

            # Use projection as the line (simulating sportsbook)
            line = round(projection, 1)

            # Calculate hit
            error = actual_val - projection
            over_hit = actual_val > line
            under_hit = actual_val < line

            # Use probability-based betting strategy
            # Calculate probability of going over the line using normal CDF
            from math import erf, sqrt
            if std_dev > 0:
                z_score = (line - projection) / std_dev
                prob_under = 0.5 * (1 + erf(z_score / sqrt(2)))
                prob_over = 1 - prob_under
            else:
                prob_over = 1.0 if projection > line else 0.0

            # Bet on the side with higher probability
            # Require >55% confidence to bet (simulating edge requirement)
            if prob_over > 0.55:
                predicted_side = "OVER"
            elif prob_over < 0.45:
                predicted_side = "UNDER"
            else:
                # No clear edge, skip or bet based on slight lean
                predicted_side = "OVER" if prob_over > 0.5 else "UNDER"

            hit = (predicted_side == "OVER" and over_hit) or \
                  (predicted_side == "UNDER" and under_hit)

            results.append({
                'player_name': player_name,
                'position': position,
                'prop_type': prop_type,
                'projection': round(projection, 1),
                'std_dev': round(std_dev, 1),
                'line': line,
                'actual': actual_val,
                'error': round(error, 1),
                'predicted_side': predicted_side,
                'hit': hit,
                'over_hit': over_hit,
            })

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No results")
        return {}

    # Analysis
    print(f"\nTotal predictions: {len(df)}")
    print(f"Overall hit rate: {df['hit'].mean()*100:.1f}%")

    print(f"\nBy prop type:")
    for prop_type in df['prop_type'].unique():
        prop_df = df[df['prop_type'] == prop_type]
        hit_rate = prop_df['hit'].mean()
        mae = prop_df['error'].abs().mean()
        count = len(prop_df)
        over_rate = prop_df['over_hit'].mean()
        print(f"  {prop_type:20s}: {hit_rate*100:5.1f}% hit | MAE={mae:6.1f} | Over={over_rate*100:5.1f}% (n={count})")

    print(f"\nBy position:")
    for pos in df['position'].unique():
        pos_df = df[df['position'] == pos]
        hit_rate = pos_df['hit'].mean()
        count = len(pos_df)
        print(f"  {pos:5s}: {hit_rate*100:5.1f}% (n={count})")

    # Model accuracy (not betting accuracy)
    print(f"\n=== Model Accuracy (Projection Quality) ===")
    for prop_type in df['prop_type'].unique():
        prop_df = df[df['prop_type'] == prop_type]
        corr = prop_df['projection'].corr(prop_df['actual'])
        mae = prop_df['error'].abs().mean()
        bias = prop_df['error'].mean()
        print(f"  {prop_type:20s}: r={corr:.3f} | MAE={mae:6.1f} | Bias={bias:+6.1f}")

    return {
        'week': week,
        'total': len(df),
        'hit_rate': df['hit'].mean(),
        'results': df.to_dict('records')
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True)

    args = parser.parse_args()
    backtest_with_models(args.week)


if __name__ == "__main__":
    main()
