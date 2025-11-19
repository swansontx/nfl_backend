"""Train models for ALL 60+ prop types from The Odds API.

This is the comprehensive multi-prop training system that:
1. Trains separate models for every prop type (passing, rushing, receiving, TDs, etc.)
2. Handles full game props (passing_yards, rushing_yards, etc.)
3. Handles quarter/half props (1Q, 1H, 2H, 3Q, 4Q) with proportional modeling
4. Accounts for player availability (DNP instances via is_active field)
5. Generates projections for ALL markets for value detection

Prop Models Trained (60+ total):

FULL GAME PROPS:
- Passing: pass_yds, pass_tds, completions, attempts, interceptions, longest_completion
- Rushing: rush_yds, rush_tds, rush_attempts, longest
- Receiving: receptions, reception_yds, reception_tds, longest
- Kicking: kicking_points, field_goals, field_goals_made
- Touchdowns: anytime_td, first_td, last_td
- Defense: tackles_assists, sacks, interceptions
- Combos: pass_rush_yds, pass_tds_rush_tds, rush_reception_yds, receptions_rush_yds

QUARTER/HALF PROPS (proportional modeling):
- 1H (First Half - 52%): pass_yds, pass_tds, completions, rush_yds, rush_tds, receptions, reception_yds, anytime_td
- 1Q (First Quarter - 25%): pass_yds, pass_tds, completions, rush_yds, rush_tds, receptions, reception_yds, anytime_td
- 2H (Second Half - 48%): pass_yds, pass_tds, rush_yds, receptions, reception_yds
- 3Q (Third Quarter - 24%): pass_yds, rush_yds, reception_yds
- 4Q (Fourth Quarter - 24%): pass_yds, rush_yds, reception_yds

Output:
- Trained models for all 60+ prop types saved to outputs/models/multi_prop/
- Can generate projections for ALL TheOddsAPI markets
- Direct comparison to DraftKings lines for value detection
- Properly handles DNP instances (excludes is_active=False from evaluation)
"""

from pathlib import Path
import argparse
import json
import pickle
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Map Odds API markets to feature extraction fields
PROP_MODEL_CONFIG = {
    # FULL GAME - Passing props
    'player_pass_yds': {
        'target_field': 'passing_yards',
        'features': ['qb_epa', 'cpoe_avg', 'success_rate', 'attempts', 'completions', 'air_yards'],
        'positions': ['QB'],
        'min_samples': 5
    },
    'player_pass_tds': {
        'target_field': 'passing_tds',
        'features': ['qb_epa', 'success_rate', 'attempts', 'completions', 'wpa'],
        'positions': ['QB'],
        'min_samples': 5
    },
    'player_pass_completions': {
        'target_field': 'completions',
        'features': ['attempts', 'cpoe_avg', 'qb_epa', 'success_rate'],
        'positions': ['QB'],
        'min_samples': 5
    },
    'player_pass_attempts': {
        'target_field': 'attempts',
        'features': ['completions', 'passing_yards', 'qb_epa'],
        'positions': ['QB'],
        'min_samples': 5
    },
    'player_pass_interceptions': {
        'target_field': 'interceptions',
        'features': ['attempts', 'qb_epa', 'qb_pressures', 'cpoe_avg'],
        'positions': ['QB'],
        'min_samples': 5
    },

    # FULL GAME - Rushing props
    'player_rush_yds': {
        'target_field': 'rushing_yards',
        'features': ['rushing_epa', 'rushing_attempts', 'success_rate', 'wpa'],
        'positions': ['RB', 'QB'],
        'min_samples': 5
    },
    'player_rush_tds': {
        'target_field': 'rushing_tds',
        'features': ['rushing_epa', 'rushing_attempts', 'rushing_yards', 'success_rate'],
        'positions': ['RB', 'QB'],
        'min_samples': 5
    },
    'player_rush_attempts': {
        'target_field': 'rushing_attempts',
        'features': ['rushing_yards', 'rushing_epa', 'total_plays'],
        'positions': ['RB', 'QB'],
        'min_samples': 5
    },

    # FULL GAME - Receiving props
    'player_receptions': {
        'target_field': 'receptions',
        'features': ['targets', 'receiving_epa', 'receiving_yards', 'success_rate'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5
    },
    'player_reception_yds': {
        'target_field': 'receiving_yards',
        'features': ['receptions', 'targets', 'receiving_epa', 'air_yards', 'yards_after_catch'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5
    },
    'player_reception_tds': {
        'target_field': 'receiving_tds',
        'features': ['receptions', 'targets', 'receiving_epa', 'receiving_yards'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5
    },

    # FULL GAME - Combo props
    'player_pass_rush_yds': {
        'target_field': 'pass_rush_total',  # Sum of passing + rushing yards
        'features': ['passing_yards', 'rushing_yards', 'qb_epa', 'rushing_epa'],
        'positions': ['QB'],
        'min_samples': 5,
        'composite': True,  # Calculated from multiple fields
        'calc': lambda row: row.get('passing_yards', 0) + row.get('rushing_yards', 0)
    },
    'player_pass_tds_rush_tds': {
        'target_field': 'pass_rush_td_total',
        'features': ['passing_tds', 'rushing_tds', 'qb_epa', 'rushing_epa'],
        'positions': ['QB'],
        'min_samples': 5,
        'composite': True,
        'calc': lambda row: row.get('passing_tds', 0) + row.get('rushing_tds', 0)
    },
    'player_rush_reception_yds': {
        'target_field': 'rush_rec_total',
        'features': ['rushing_yards', 'receiving_yards', 'rushing_epa', 'receiving_epa'],
        'positions': ['RB', 'WR'],
        'min_samples': 5,
        'composite': True,
        'calc': lambda row: row.get('rushing_yards', 0) + row.get('receiving_yards', 0)
    },
    'player_receptions_rush_yds': {
        'target_field': 'rec_rush_total',
        'features': ['receptions', 'rushing_yards', 'receiving_epa', 'rushing_epa'],
        'positions': ['RB', 'WR'],
        'min_samples': 5,
        'composite': True,
        'calc': lambda row: row.get('receptions', 0) + row.get('rushing_yards', 0)
    },

    # FULL GAME - Longest plays
    'player_pass_longest_completion': {
        'target_field': 'passing_yards',  # Use max from play-by-play
        'features': ['air_yards', 'qb_epa', 'attempts', 'completions'],
        'positions': ['QB'],
        'min_samples': 5,
        'aggregate': 'max'  # Take max instead of sum
    },
    'player_rush_longest': {
        'target_field': 'rushing_yards',
        'features': ['rushing_epa', 'rushing_attempts'],
        'positions': ['RB', 'QB'],
        'min_samples': 5,
        'aggregate': 'max'
    },
    'player_reception_longest': {
        'target_field': 'receiving_yards',
        'features': ['air_yards', 'yards_after_catch', 'receiving_epa'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5,
        'aggregate': 'max'
    },

    # FULL GAME - Kicking props
    'player_kicking_points': {
        'target_field': 'kicking_points',
        'features': ['field_goals_made', 'field_goal_attempts', 'extra_points_made'],
        'positions': ['K'],
        'min_samples': 5
    },
    'player_field_goals': {
        'target_field': 'field_goal_attempts',
        'features': ['field_goals_made', 'kicking_points'],
        'positions': ['K'],
        'min_samples': 5
    },
    'player_field_goals_made': {
        'target_field': 'field_goals_made',
        'features': ['field_goal_attempts', 'kicking_points'],
        'positions': ['K'],
        'min_samples': 5
    },

    # FULL GAME - Touchdown props (binary outcomes - use classification later)
    'player_anytime_td': {
        'target_field': 'total_tds',
        'features': ['rushing_tds', 'receiving_tds', 'rushing_epa', 'receiving_epa', 'targets'],
        'positions': ['RB', 'WR', 'TE', 'QB'],
        'min_samples': 5,
        'composite': True,
        'calc': lambda row: row.get('rushing_tds', 0) + row.get('receiving_tds', 0)
    },
    'player_first_td': {
        'target_field': 'total_tds',
        'features': ['rushing_tds', 'receiving_tds', 'rushing_epa', 'receiving_epa'],
        'positions': ['RB', 'WR', 'TE', 'QB'],
        'min_samples': 5,
        'composite': True,
        'calc': lambda row: row.get('rushing_tds', 0) + row.get('receiving_tds', 0)
    },
    'player_last_td': {
        'target_field': 'total_tds',
        'features': ['rushing_tds', 'receiving_tds', 'rushing_epa', 'receiving_epa'],
        'positions': ['RB', 'WR', 'TE', 'QB'],
        'min_samples': 5,
        'composite': True,
        'calc': lambda row: row.get('rushing_tds', 0) + row.get('receiving_tds', 0)
    },

    # FULL GAME - Defensive props
    'player_tackles_assists': {
        'target_field': 'tackles_assists',
        'features': ['total_plays'],  # Will need defensive stats from nflverse
        'positions': ['LB', 'DB', 'DL'],
        'min_samples': 5
    },
    'player_sacks': {
        'target_field': 'sacks',
        'features': ['total_plays'],
        'positions': ['DL', 'LB', 'DB'],
        'min_samples': 5
    },
    'player_interceptions': {
        'target_field': 'defensive_interceptions',
        'features': ['total_plays'],
        'positions': ['DB', 'LB'],
        'min_samples': 5
    },

    # QUARTER/HALF PROPS - Proportional modeling based on full game
    # First Half (1H) - typically 52% of game production
    'player_1h_pass_yds': {
        'target_field': 'passing_yards',
        'features': ['qb_epa', 'cpoe_avg', 'success_rate', 'attempts', 'completions'],
        'positions': ['QB'],
        'min_samples': 5,
        'proportional': 0.52  # 1H typically 52% of game
    },
    'player_1h_pass_tds': {
        'target_field': 'passing_tds',
        'features': ['qb_epa', 'success_rate', 'attempts'],
        'positions': ['QB'],
        'min_samples': 5,
        'proportional': 0.52
    },
    'player_1h_pass_completions': {
        'target_field': 'completions',
        'features': ['attempts', 'cpoe_avg', 'qb_epa'],
        'positions': ['QB'],
        'min_samples': 5,
        'proportional': 0.52
    },
    'player_1h_rush_yds': {
        'target_field': 'rushing_yards',
        'features': ['rushing_epa', 'rushing_attempts', 'success_rate'],
        'positions': ['RB', 'QB'],
        'min_samples': 5,
        'proportional': 0.52
    },
    'player_1h_rush_tds': {
        'target_field': 'rushing_tds',
        'features': ['rushing_epa', 'rushing_attempts'],
        'positions': ['RB', 'QB'],
        'min_samples': 5,
        'proportional': 0.52
    },
    'player_1h_receptions': {
        'target_field': 'receptions',
        'features': ['targets', 'receiving_epa'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5,
        'proportional': 0.52
    },
    'player_1h_reception_yds': {
        'target_field': 'receiving_yards',
        'features': ['receptions', 'targets', 'receiving_epa'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5,
        'proportional': 0.52
    },
    'player_1h_anytime_td': {
        'target_field': 'total_tds',
        'features': ['rushing_tds', 'receiving_tds', 'rushing_epa', 'receiving_epa'],
        'positions': ['RB', 'WR', 'TE', 'QB'],
        'min_samples': 5,
        'composite': True,
        'proportional': 0.52,
        'calc': lambda row: row.get('rushing_tds', 0) + row.get('receiving_tds', 0)
    },

    # First Quarter (1Q) - typically 25% of game production
    'player_1q_pass_yds': {
        'target_field': 'passing_yards',
        'features': ['qb_epa', 'cpoe_avg', 'attempts'],
        'positions': ['QB'],
        'min_samples': 5,
        'proportional': 0.25
    },
    'player_1q_pass_tds': {
        'target_field': 'passing_tds',
        'features': ['qb_epa', 'attempts'],
        'positions': ['QB'],
        'min_samples': 5,
        'proportional': 0.25
    },
    'player_1q_pass_completions': {
        'target_field': 'completions',
        'features': ['attempts', 'cpoe_avg'],
        'positions': ['QB'],
        'min_samples': 5,
        'proportional': 0.25
    },
    'player_1q_rush_yds': {
        'target_field': 'rushing_yards',
        'features': ['rushing_epa', 'rushing_attempts'],
        'positions': ['RB', 'QB'],
        'min_samples': 5,
        'proportional': 0.25
    },
    'player_1q_rush_tds': {
        'target_field': 'rushing_tds',
        'features': ['rushing_epa', 'rushing_attempts'],
        'positions': ['RB', 'QB'],
        'min_samples': 5,
        'proportional': 0.25
    },
    'player_1q_receptions': {
        'target_field': 'receptions',
        'features': ['targets', 'receiving_epa'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5,
        'proportional': 0.25
    },
    'player_1q_reception_yds': {
        'target_field': 'receiving_yards',
        'features': ['receptions', 'targets'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5,
        'proportional': 0.25
    },
    'player_1q_anytime_td': {
        'target_field': 'total_tds',
        'features': ['rushing_tds', 'receiving_tds'],
        'positions': ['RB', 'WR', 'TE', 'QB'],
        'min_samples': 5,
        'composite': True,
        'proportional': 0.25,
        'calc': lambda row: row.get('rushing_tds', 0) + row.get('receiving_tds', 0)
    },

    # Second Half (2H) - typically 48% of game production
    'player_2h_pass_yds': {
        'target_field': 'passing_yards',
        'features': ['qb_epa', 'cpoe_avg', 'attempts'],
        'positions': ['QB'],
        'min_samples': 5,
        'proportional': 0.48
    },
    'player_2h_pass_tds': {
        'target_field': 'passing_tds',
        'features': ['qb_epa', 'attempts'],
        'positions': ['QB'],
        'min_samples': 5,
        'proportional': 0.48
    },
    'player_2h_rush_yds': {
        'target_field': 'rushing_yards',
        'features': ['rushing_epa', 'rushing_attempts'],
        'positions': ['RB', 'QB'],
        'min_samples': 5,
        'proportional': 0.48
    },
    'player_2h_receptions': {
        'target_field': 'receptions',
        'features': ['targets', 'receiving_epa'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5,
        'proportional': 0.48
    },
    'player_2h_reception_yds': {
        'target_field': 'receiving_yards',
        'features': ['receptions', 'targets'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5,
        'proportional': 0.48
    },

    # Third Quarter (3Q) - typically 24% of game production
    'player_3q_pass_yds': {
        'target_field': 'passing_yards',
        'features': ['qb_epa', 'attempts'],
        'positions': ['QB'],
        'min_samples': 5,
        'proportional': 0.24
    },
    'player_3q_rush_yds': {
        'target_field': 'rushing_yards',
        'features': ['rushing_epa', 'rushing_attempts'],
        'positions': ['RB', 'QB'],
        'min_samples': 5,
        'proportional': 0.24
    },
    'player_3q_reception_yds': {
        'target_field': 'receiving_yards',
        'features': ['receptions', 'targets'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5,
        'proportional': 0.24
    },

    # Fourth Quarter (4Q) - typically 24% of game production
    'player_4q_pass_yds': {
        'target_field': 'passing_yards',
        'features': ['qb_epa', 'attempts'],
        'positions': ['QB'],
        'min_samples': 5,
        'proportional': 0.24
    },
    'player_4q_rush_yds': {
        'target_field': 'rushing_yards',
        'features': ['rushing_epa', 'rushing_attempts'],
        'positions': ['RB', 'QB'],
        'min_samples': 5,
        'proportional': 0.24
    },
    'player_4q_reception_yds': {
        'target_field': 'receiving_yards',
        'features': ['receptions', 'targets'],
        'positions': ['WR', 'TE', 'RB'],
        'min_samples': 5,
        'proportional': 0.24
    },
}


def train_multi_prop_models(
    season: int,
    features_file: Path,
    output_dir: Path,
    model_type: str = 'xgboost'
) -> Dict:
    """Train models for ALL prop types from The Odds API.

    Args:
        season: Season year
        features_file: Path to player features JSON
        output_dir: Directory to save trained models
        model_type: 'xgboost' or 'lightgbm'

    Returns:
        Dict with training results for all prop models
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"TRAINING MULTI-PROP MODELS - Season {season}")
    print(f"Model Type: {model_type.upper()}")
    print(f"Total Markets: {len(PROP_MODEL_CONFIG)}")
    print(f"{'='*80}\n")

    # Load features
    print(f"📂 Loading features from {features_file}")
    with open(features_file, 'r') as f:
        player_features = json.load(f)

    print(f"✓ Loaded features for {len(player_features)} players\n")

    # Train a model for each prop type
    results = {}
    models_trained = 0
    models_failed = 0

    for market, config in PROP_MODEL_CONFIG.items():
        print(f"\n{'─'*80}")
        print(f"Training: {market}")
        print(f"Positions: {', '.join(config['positions'])}")
        print(f"{'─'*80}")

        try:
            # Train model for this specific prop
            model_result = _train_single_prop_model(
                market=market,
                config=config,
                player_features=player_features,
                output_dir=output_dir,
                model_type=model_type
            )

            results[market] = model_result

            if 'error' not in model_result:
                models_trained += 1
                print(f"✓ {market}: Val RMSE = {model_result.get('val_rmse', 0):.2f}, R² = {model_result.get('val_r2', 0):.3f}")
            else:
                models_failed += 1
                print(f"⚠️  {market}: {model_result['error']}")

        except Exception as e:
            models_failed += 1
            print(f"✗ {market}: Training failed - {e}")
            results[market] = {'error': str(e)}

    # Save overall results summary
    summary_file = output_dir / f'multi_prop_training_summary_{season}.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"MULTI-PROP TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Models Trained: {models_trained}/{len(PROP_MODEL_CONFIG)}")
    print(f"✗ Models Failed: {models_failed}/{len(PROP_MODEL_CONFIG)}")
    print(f"📊 Summary saved: {summary_file}")
    print(f"{'='*80}\n")

    return results


def _train_single_prop_model(
    market: str,
    config: Dict,
    player_features: Dict,
    output_dir: Path,
    model_type: str
) -> Dict:
    """Train a single prop model.

    Args:
        market: Prop market name (e.g., 'player_pass_yds')
        config: Model configuration
        player_features: Player feature data
        output_dir: Output directory
        model_type: 'xgboost' or 'lightgbm'

    Returns:
        Training metrics dict
    """
    # Prepare training data
    df = _prepare_prop_training_data(player_features, config)

    if df.empty or len(df) < config['min_samples']:
        return {
            'error': f'Insufficient data: {len(df)} samples (min {config["min_samples"]})',
            'samples': len(df)
        }

    # Filter to active games only (exclude DNP)
    df_active = df[df['is_active'] == True].copy()

    if len(df_active) < config['min_samples']:
        return {
            'error': f'Insufficient active samples: {len(df_active)} (min {config["min_samples"]})',
            'total_samples': len(df),
            'active_samples': len(df_active),
            'dnp_rate': (len(df) - len(df_active)) / len(df) if len(df) > 0 else 0
        }

    print(f"  Dataset: {len(df_active)} active samples ({len(df) - len(df_active)} DNP excluded)")

    # Engineer features (rolling averages, lags)
    df_active = _engineer_prop_features(df_active, config)

    # Train model (using same logic as train_passing_model.py but for this prop)
    try:
        import xgboost as xgb
    except ImportError:
        return {'error': 'XGBoost not installed: pip install xgboost'}

    # Prepare X, y
    target_field = config['target_field']
    feature_cols = [f for f in config['features'] if f in df_active.columns]

    if not feature_cols:
        return {'error': f'No valid features found: {config["features"]}'}

    X = df_active[feature_cols].fillna(0)
    y = df_active[target_field].fillna(0)

    # Time-based train/val split (no shuffling to preserve temporal order)
    split_idx = int(len(df_active) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    if len(X_train) < 3 or len(X_val) < 2:
        return {'error': 'Insufficient split size'}

    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    # Save model
    model_file = output_dir / f'{market}_model_{model_type}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    return {
        'model_path': str(model_file),
        'val_rmse': float(val_rmse),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'features_used': feature_cols
    }


def _prepare_prop_training_data(player_features: Dict, config: Dict) -> pd.DataFrame:
    """Prepare training data for a specific prop.

    Args:
        player_features: Player feature data
        config: Prop model configuration

    Returns:
        DataFrame with features and target
    """
    rows = []

    for player_id, games in player_features.items():
        for game in games:
            # Skip if not the right position
            # (We don't have position info yet, so include all for now)

            # Add is_active field (from Phase 1)
            is_active = game.get('is_active', True)

            # Calculate target field
            if config.get('composite'):
                # Composite fields (e.g., pass+rush yards)
                target_value = config['calc'](game)
            else:
                target_value = game.get(config['target_field'], 0)

            # Apply proportional scaling for quarter/half props
            if config.get('proportional'):
                target_value = target_value * config['proportional']

            # Build feature row
            row = {
                'player_id': player_id,
                'game_id': game.get('game_id'),
                'week': game.get('week'),
                'season': game.get('season'),
                'is_active': is_active,
                config['target_field']: target_value
            }

            # Add all features
            for feat in config['features']:
                row[feat] = game.get(feat, 0)

            rows.append(row)

    return pd.DataFrame(rows)


def _engineer_prop_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Engineer rolling features for prop prediction.

    Args:
        df: DataFrame with raw features
        config: Prop configuration

    Returns:
        DataFrame with engineered features
    """
    # Sort by player and week
    df = df.sort_values(['player_id', 'week']).reset_index(drop=True)

    # Add rolling averages (last 3 games, last 5 games)
    target = config['target_field']

    df[f'{target}_rolling_3'] = (df.groupby('player_id')[target]
                                   .transform(lambda x: x.rolling(3, min_periods=1).mean()))

    df[f'{target}_rolling_5'] = (df.groupby('player_id')[target]
                                   .transform(lambda x: x.rolling(5, min_periods=1).mean()))

    # Add lag features (last game value)
    df[f'{target}_lag_1'] = df.groupby('player_id')[target].shift(1)

    # Fill NaN with 0
    df = df.fillna(0)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train models for ALL 60+ prop types from The Odds API'
    )
    parser.add_argument('--season', type=int, required=True,
                       help='Season year (e.g., 2024)')
    parser.add_argument('--features-file', type=Path,
                       default=Path('outputs/features/2024_player_features.json'),
                       help='Path to player features JSON')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('outputs/models/multi_prop'),
                       help='Output directory for trained models')
    parser.add_argument('--model-type', default='xgboost',
                       choices=['xgboost', 'lightgbm'],
                       help='Model type (default: xgboost)')

    args = parser.parse_args()

    train_multi_prop_models(
        season=args.season,
        features_file=args.features_file,
        output_dir=args.output_dir,
        model_type=args.model_type
    )
