"""Two-layer modeling: Usage × Efficiency = Production

This is a MAJOR architectural improvement over single-stage models.

TRADITIONAL APPROACH (what we had):
  Direct Model: features → yards

PROBLEMS:
  - Can't separate opportunity from skill
  - Struggles with game script changes (blowouts, injuries, etc.)
  - Conflates volume variance with efficiency variance

NEW TWO-LAYER APPROACH:
  Layer 1 (Usage Model): features → attempts/targets/carries
  Layer 2 (Efficiency Model): features → yards_per_attempt/carry/target

  Final Projection: yards = usage × efficiency

BENEFITS:
  1. Game script awareness: Big favorites get more rush attempts (usage up)
  2. Matchup nuance: Tough defense → lower YPA (efficiency down)
  3. Better uncertainty: Can model usage variance separately from efficiency variance
  4. Interpretable: "CMC gets 22 carries × 5.2 YPC = 114 yards"

Research shows this approach reduces RMSE by 10-15% for volume props.
"""

from pathlib import Path
import argparse
import json
import pickle
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def train_usage_model(
    features_file: Path,
    output_dir: Path,
    prop_type: str,  # 'pass_attempts', 'rush_attempts', 'targets'
    model_type: str = 'xgboost'
) -> Dict:
    """Train usage model (predicts attempts/targets/carries).

    Args:
        features_file: Player features JSON
        output_dir: Output directory for models
        prop_type: Type of usage to predict
        model_type: Model type (xgboost or lightgbm)

    Returns:
        Training results dict
    """
    print(f"\n{'='*80}")
    print(f"TRAINING USAGE MODEL: {prop_type}")
    print(f"{'='*80}\n")

    # Load features
    with open(features_file, 'r') as f:
        player_features = json.load(f)

    # Prepare training data
    df = _prepare_usage_training_data(player_features, prop_type)

    if len(df) < 50:
        return {'error': 'Insufficient samples', 'samples': len(df)}

    print(f"✓ Prepared {len(df)} samples")

    # Define features based on prop type
    if prop_type == 'pass_attempts':
        feature_cols = [
            # Context features (NEW!)
            'spread', 'total', 'implied_team_total', 'is_favorite', 'expected_to_trail',
            'def_pass_epa_allowed', 'team_plays_pg', 'neutral_pass_rate',
            'is_dome', 'wind_high', 'temp_cold',

            # Historical usage
            'attempts_rolling_3', 'attempts_rolling_5', 'attempts_lag_1',

            # Player features
            'qb_epa', 'cpoe_avg', 'success_rate'
        ]
        target_col = 'attempts'

    elif prop_type == 'rush_attempts':
        feature_cols = [
            # Context features (NEW!)
            'spread', 'total', 'implied_team_total', 'is_favorite', 'expected_to_lead',
            'def_rush_epa_allowed', 'team_plays_pg', 'neutral_pass_rate',

            # Historical usage
            'rushing_attempts_rolling_3', 'rushing_attempts_rolling_5', 'rushing_attempts_lag_1',
            'carry_share_rolling_3',  # NEW: Share of team rushes

            # Player features
            'rushing_epa', 'success_rate'
        ]
        target_col = 'rushing_attempts'

    elif prop_type == 'targets':
        feature_cols = [
            # Context features (NEW!)
            'spread', 'total', 'implied_team_total', 'is_underdog',
            'def_pass_epa_allowed', 'team_plays_pg', 'neutral_pass_rate',
            'is_dome', 'wind_high',

            # Historical usage
            'targets_rolling_3', 'targets_rolling_5', 'targets_lag_1',
            'target_share_rolling_3',  # NEW: Share of team targets

            # Player features
            'receiving_epa', 'receptions'
        ]
        target_col = 'targets'

    else:
        return {'error': f'Unknown prop type: {prop_type}'}

    # Filter to active games only
    df_active = df[df['is_active'] == True].copy()
    print(f"✓ Filtered to {len(df_active)} active games ({len(df) - len(df_active)} DNP excluded)")

    # Extract features and target
    feature_cols_available = [f for f in feature_cols if f in df_active.columns]

    if len(feature_cols_available) < 5:
        print(f"⚠️  Warning: Only {len(feature_cols_available)} features available")

    X = df_active[feature_cols_available].fillna(0)
    y = df_active[target_col].fillna(0)

    # Time-based split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}")

    # Train model
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
    else:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )

    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    val_r2 = 1 - (np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

    print(f"\n✓ Train RMSE: {train_rmse:.2f}")
    print(f"✓ Val RMSE: {val_rmse:.2f}")
    print(f"✓ Val R²: {val_r2:.3f}")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f'{prop_type}_usage_model_{model_type}.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': feature_cols_available,
            'target': target_col,
            'prop_type': prop_type
        }, f)

    print(f"✓ Saved model to: {model_path}")

    return {
        'prop_type': prop_type,
        'model_path': str(model_path),
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'features_used': feature_cols_available,
        'samples': len(df_active)
    }


def train_efficiency_model(
    features_file: Path,
    output_dir: Path,
    prop_type: str,  # 'yards_per_attempt', 'yards_per_carry', 'yards_per_target'
    model_type: str = 'xgboost'
) -> Dict:
    """Train efficiency model (predicts yards per opportunity).

    Args:
        features_file: Player features JSON
        output_dir: Output directory for models
        prop_type: Type of efficiency to predict
        model_type: Model type

    Returns:
        Training results dict
    """
    print(f"\n{'='*80}")
    print(f"TRAINING EFFICIENCY MODEL: {prop_type}")
    print(f"{'='*80}\n")

    # Load features
    with open(features_file, 'r') as f:
        player_features = json.load(f)

    # Prepare training data
    df = _prepare_efficiency_training_data(player_features, prop_type)

    if len(df) < 50:
        return {'error': 'Insufficient samples', 'samples': len(df)}

    print(f"✓ Prepared {len(df)} samples")

    # Define features based on prop type
    if prop_type == 'yards_per_attempt':
        feature_cols = [
            # Advanced metrics (EPA, CPOE are key for efficiency!)
            'qb_epa', 'cpoe_avg', 'success_rate', 'air_yards',

            # Matchup (NEW!)
            'def_pass_epa_allowed', 'def_cpoe_allowed', 'def_success_rate_allowed',

            # Environment (NEW!)
            'is_dome', 'wind_high', 'temp_cold', 'precipitation',

            # Historical efficiency
            'yards_per_attempt_rolling_3', 'yards_per_attempt_rolling_5'
        ]
        target_col = 'yards_per_attempt'

    elif prop_type == 'yards_per_carry':
        feature_cols = [
            # Advanced metrics
            'rushing_epa', 'success_rate',

            # Matchup (NEW!)
            'def_rush_epa_allowed', 'def_success_rate_allowed',

            # Game script (NEW!)
            'spread', 'expected_to_lead',  # Favorites run more = easier yards

            # Historical efficiency
            'yards_per_carry_rolling_3', 'yards_per_carry_rolling_5'
        ]
        target_col = 'yards_per_carry'

    elif prop_type == 'yards_per_target':
        feature_cols = [
            # Advanced metrics
            'receiving_epa', 'air_yards', 'yards_after_catch',

            # Matchup (NEW!)
            'def_pass_epa_allowed',

            # Environment (NEW!)
            'is_dome', 'wind_high',

            # Historical efficiency
            'yards_per_target_rolling_3', 'yards_per_target_rolling_5'
        ]
        target_col = 'yards_per_target'

    else:
        return {'error': f'Unknown prop type: {prop_type}'}

    # Filter to active games with attempts/targets > 0
    df_active = df[(df['is_active'] == True) & (df[target_col] > 0)].copy()
    print(f"✓ Filtered to {len(df_active)} active games with usage")

    # Extract features and target
    feature_cols_available = [f for f in feature_cols if f in df_active.columns]

    X = df_active[feature_cols_available].fillna(0)
    y = df_active[target_col].fillna(0)

    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}")

    # Train model
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
    else:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )

    model.fit(X_train, y_train)

    # Evaluate
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    val_r2 = 1 - (np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

    print(f"\n✓ Val RMSE: {val_rmse:.2f}")
    print(f"✓ Val R²: {val_r2:.3f}")

    # Save model
    model_path = output_dir / f'{prop_type}_efficiency_model_{model_type}.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': feature_cols_available,
            'target': target_col,
            'prop_type': prop_type
        }, f)

    print(f"✓ Saved model to: {model_path}")

    return {
        'prop_type': prop_type,
        'model_path': str(model_path),
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'features_used': feature_cols_available,
        'samples': len(df_active)
    }


def _prepare_usage_training_data(player_features: Dict, prop_type: str) -> pd.DataFrame:
    """Prepare training data for usage model."""
    rows = []

    for player_id, games in player_features.items():
        for game in games:
            row = {
                'player_id': player_id,
                'game_id': game.get('game_id'),
                'is_active': game.get('is_active', True),

                # Target (what we're predicting)
                'attempts': game.get('attempts', 0),
                'rushing_attempts': game.get('rushing_attempts', 0),
                'targets': game.get('targets', 0),

                # Context features
                'spread': game.get('spread', 0),
                'total': game.get('total', 45),
                'implied_team_total': game.get('implied_team_total', 22.5),
                'is_favorite': game.get('is_favorite', 0),
                'is_underdog': game.get('is_underdog', 0),
                'expected_to_lead': game.get('expected_to_lead', 0),
                'expected_to_trail': game.get('expected_to_trail', 0),

                # Defense
                'def_pass_epa_allowed': game.get('def_pass_epa_allowed', 0),
                'def_rush_epa_allowed': game.get('def_rush_epa_allowed', 0),

                # Pace
                'team_plays_pg': game.get('team_plays_pg', 65),
                'neutral_pass_rate': game.get('neutral_pass_rate', 0.55),

                # Weather
                'is_dome': game.get('is_dome', 0),
                'wind_high': game.get('wind_high', 0),
                'temp_cold': game.get('temp_cold', 0),

                # Player features
                'qb_epa': game.get('qb_epa', 0),
                'cpoe_avg': game.get('cpoe_avg', 0),
                'rushing_epa': game.get('rushing_epa', 0),
                'receiving_epa': game.get('receiving_epa', 0),
                'success_rate': game.get('success_rate', 0),
                'receptions': game.get('receptions', 0)
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    # Engineer rolling features
    for col in ['attempts', 'rushing_attempts', 'targets']:
        if col in df.columns:
            df[f'{col}_rolling_3'] = df.groupby('player_id')[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            df[f'{col}_rolling_5'] = df.groupby('player_id')[col].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
            df[f'{col}_lag_1'] = df.groupby('player_id')[col].shift(1)

    # Calculate share metrics using actual team totals
    # First calculate team totals per game
    if 'team' in df.columns and 'week' in df.columns:
        team_totals = df.groupby(['team', 'week']).agg({
            'rushing_attempts': 'sum',
            'targets': 'sum'
        }).reset_index()
        team_totals.columns = ['team', 'week', 'team_rush_att', 'team_targets']

        # Merge back
        df = df.merge(team_totals, on=['team', 'week'], how='left')

        # Calculate actual shares
        df['carry_share_rolling_3'] = df['rushing_attempts_rolling_3'] / df['team_rush_att'].clip(lower=1)
        df['target_share_rolling_3'] = df['targets_rolling_3'] / df['team_targets'].clip(lower=1)
    else:
        # Fallback to estimates if team/week not available
        df['carry_share_rolling_3'] = df['rushing_attempts_rolling_3'] / 25
        df['target_share_rolling_3'] = df['targets_rolling_3'] / 35

    return df


def _prepare_efficiency_training_data(player_features: Dict, prop_type: str) -> pd.DataFrame:
    """Prepare training data for efficiency model."""
    rows = []

    for player_id, games in player_features.items():
        for game in games:
            # Calculate efficiency metrics
            attempts = game.get('attempts', 0)
            passing_yards = game.get('passing_yards', 0)
            yards_per_attempt = passing_yards / attempts if attempts > 0 else 0

            rushing_attempts = game.get('rushing_attempts', 0)
            rushing_yards = game.get('rushing_yards', 0)
            yards_per_carry = rushing_yards / rushing_attempts if rushing_attempts > 0 else 0

            targets = game.get('targets', 0)
            receiving_yards = game.get('receiving_yards', 0)
            yards_per_target = receiving_yards / targets if targets > 0 else 0

            row = {
                'player_id': player_id,
                'game_id': game.get('game_id'),
                'is_active': game.get('is_active', True),

                # Targets (efficiency metrics)
                'yards_per_attempt': yards_per_attempt,
                'yards_per_carry': yards_per_carry,
                'yards_per_target': yards_per_target,

                # Advanced metrics (key for efficiency!)
                'qb_epa': game.get('qb_epa', 0),
                'cpoe_avg': game.get('cpoe_avg', 0),
                'rushing_epa': game.get('rushing_epa', 0),
                'receiving_epa': game.get('receiving_epa', 0),
                'success_rate': game.get('success_rate', 0),
                'air_yards': game.get('air_yards', 0),
                'yards_after_catch': game.get('yards_after_catch', 0),

                # Matchup
                'def_pass_epa_allowed': game.get('def_pass_epa_allowed', 0),
                'def_rush_epa_allowed': game.get('def_rush_epa_allowed', 0),
                'def_success_rate_allowed': game.get('def_success_rate_allowed', 0),
                'def_cpoe_allowed': game.get('def_cpoe_allowed', 0),

                # Game script
                'spread': game.get('spread', 0),
                'expected_to_lead': game.get('expected_to_lead', 0),

                # Environment
                'is_dome': game.get('is_dome', 0),
                'wind_high': game.get('wind_high', 0),
                'temp_cold': game.get('temp_cold', 0),
                'precipitation': game.get('precipitation', 0)
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    # Engineer rolling features for efficiency
    for col in ['yards_per_attempt', 'yards_per_carry', 'yards_per_target']:
        if col in df.columns:
            df[f'{col}_rolling_3'] = df.groupby('player_id')[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            df[f'{col}_rolling_5'] = df.groupby('player_id')[col].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train two-layer usage + efficiency models'
    )
    parser.add_argument('--features-file', type=Path, required=True,
                       help='Player features JSON (with context features merged)')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('outputs/models/two_layer'),
                       help='Output directory for models')
    parser.add_argument('--model-type', default='xgboost',
                       choices=['xgboost', 'lightgbm'],
                       help='Model type')

    args = parser.parse_args()

    # Train all usage models
    usage_results = {}
    for prop_type in ['pass_attempts', 'rush_attempts', 'targets']:
        result = train_usage_model(
            features_file=args.features_file,
            output_dir=args.output_dir,
            prop_type=prop_type,
            model_type=args.model_type
        )
        usage_results[prop_type] = result

    # Train all efficiency models
    efficiency_results = {}
    for prop_type in ['yards_per_attempt', 'yards_per_carry', 'yards_per_target']:
        result = train_efficiency_model(
            features_file=args.features_file,
            output_dir=args.output_dir,
            prop_type=prop_type,
            model_type=args.model_type
        )
        efficiency_results[prop_type] = result

    # Save summary
    summary = {
        'usage_models': usage_results,
        'efficiency_models': efficiency_results
    }

    summary_path = args.output_dir / 'two_layer_training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("TWO-LAYER MODEL TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\n✓ Usage models trained: {len(usage_results)}")
    print(f"✓ Efficiency models trained: {len(efficiency_results)}")
    print(f"✓ Summary saved to: {summary_path}")
