"""Train passing prop prediction model with ADVANCED METRICS and MULTI-SEASON support.

Trains XGBoost model to predict QB passing props using EPA, CPOE, and success rate.
Now supports training on multiple seasons with recency weighting for better performance.

Features used:
- EPA-based metrics: qb_epa, total_epa (rolling averages)
- CPOE (Completion % Over Expected): QB accuracy metric
- Success Rate: percentage of positive EPA plays
- Air Yards & YAC: passing value decomposition
- WPA (Win Probability Added): clutch performance
- QB Pressure: hits, hurries, sacks faced
- Traditional stats: yards, TDs, completions (for context)
- Rolling windows: 3-game, 5-game, season averages

Multi-Season Training:
- Combines data from multiple seasons for robust learning
- Recency weighting: Recent seasons weighted higher (exp decay)
- Default: 2020-2025 with decay=0.3 (2025: 1.00, 2024: 0.74, 2023: 0.55, etc.)

Model Types:
- XGBoost (primary) - Best for tabular NFL data
- LightGBM (alternative) - Faster training, similar performance

Output:
- Trained model saved to outputs/models/passing_model.pkl
- Feature importance analysis (JSON + visualization)
- Training metrics (RMSE, MAE, R², feature importance)

Usage:
    # Multi-season training (recommended):
    python -m backend.modeling.train_passing_model --seasons 2020 2021 2022 2023 2024 2025

    # Single season (backwards compatible):
    python -m backend.modeling.train_passing_model --season 2024

    # Disable recency weighting:
    python -m backend.modeling.train_passing_model --seasons 2020 2021 2022 2023 2024 2025 --no-recency-weighting

    # Custom decay rate:
    python -m backend.modeling.train_passing_model --seasons 2020 2021 2022 2023 2024 2025 --recency-decay 0.4
"""

from pathlib import Path
import argparse
import json
import pickle
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict


def train_passing_model(
    seasons: List[int],
    features_file: Path,
    output_model_path: Path,
    model_type: str = 'xgboost',
    use_recency_weighting: bool = True,
    recency_decay: float = 0.3
) -> Dict:
    """Train passing prop prediction model with advanced metrics.

    Args:
        seasons: List of season years to train on (e.g., [2020, 2021, 2022, 2023, 2024, 2025])
        features_file: Path to player features JSON (from extract_player_pbp_features.py)
        output_model_path: Path to save trained model
        model_type: 'xgboost' or 'lightgbm'
        use_recency_weighting: If True, weight recent seasons higher (default: True)
        recency_decay: Decay rate for exponential weighting (default: 0.3)

    Returns:
        Training metrics dictionary with RMSE, MAE, R², feature importance
    """
    # Convert single season to list for backwards compatibility
    if isinstance(seasons, int):
        seasons = [seasons]

    print(f"\n{'='*60}")
    print(f"Training Passing Model - Seasons {min(seasons)}-{max(seasons)}")
    print(f"Model Type: {model_type.upper()}")
    if use_recency_weighting:
        print(f"Recency Weighting: ENABLED (decay={recency_decay})")
    print(f"{'='*60}\n")

    # Check if features file exists
    if not features_file.exists():
        print(f"⚠️  Features file not found: {features_file}")
        print(f"Run feature extraction first:")
        print(f"  python -m backend.features.extract_player_pbp_features")
        return {
            'error': 'Features file not found',
            'model_type': model_type,
            'season': season
        }

    # STEP 1: Load features
    print(f"📂 Loading features from {features_file}")
    with open(features_file, 'r') as f:
        player_features = json.load(f)

    print(f"✓ Loaded features for {len(player_features)} players")

    # STEP 2: Convert to DataFrame and prepare dataset
    print(f"\n📊 Preparing training dataset...")
    df = _prepare_training_data(player_features)

    if df.empty:
        print(f"⚠️  No training data available")
        return {
            'error': 'No training data',
            'model_type': model_type,
            'seasons': seasons
        }

    print(f"✓ Dataset prepared: {len(df)} samples")
    print(f"  QB samples: {len(df[df['is_qb'] == 1])}")
    print(f"  Features: {len(df.columns)} columns")

    # STEP 2b: Calculate recency weights if enabled
    if use_recency_weighting and 'season' in df.columns:
        print(f"\n⚖️  Calculating recency weights...")
        df['sample_weight'] = _calculate_recency_weights(df['season'], recency_decay)

        # Show weight distribution
        weight_by_season = df.groupby('season')['sample_weight'].first().sort_index()
        print(f"  Season weights:")
        for season_year, weight in weight_by_season.items():
            season_samples = len(df[df['season'] == season_year])
            print(f"    {season_year}: {weight:.3f} ({season_samples} samples)")
    else:
        # Uniform weights
        df['sample_weight'] = 1.0

    # STEP 3: Feature engineering (rolling averages, lag features)
    print(f"\n⚙️  Engineering rolling features...")
    df = _engineer_rolling_features(df)

    # STEP 4: Filter to QBs only with sufficient data
    print(f"\n🎯 Filtering to QB samples with sufficient history...")
    df = df[
        (df['is_qb'] == 1) &
        (df['games_played'] >= 3) &  # At least 3 games of history
        (df['attempts'] >= 15)  # At least 15 attempts in game
    ].copy()

    print(f"✓ {len(df)} QB samples ready for training")

    if len(df) < 50:
        print(f"⚠️  Insufficient data for training (need at least 50 samples)")
        return {
            'error': 'Insufficient training samples',
            'model_type': model_type,
            'season': season,
            'num_samples': len(df)
        }

    # STEP 5: Define features and target
    feature_cols = [
        # Advanced metrics (primary features)
        'qb_epa_avg', 'total_epa_avg', 'cpoe_avg', 'success_rate',
        'wpa_avg', 'air_epa_avg', 'yac_epa_avg',
        'qb_pressure_rate', 'qb_hit_rate',

        # Rolling windows
        'qb_epa_roll_3', 'qb_epa_roll_5',
        'cpoe_roll_3', 'cpoe_roll_5',
        'success_rate_roll_3', 'success_rate_roll_5',

        # Traditional stats (context)
        'passing_yards_avg', 'completion_pct',
        'yards_per_attempt', 'air_yards_avg',

        # Game context
        'games_played', 'attempts'
    ]

    # Target: next game passing yards
    target_col = 'passing_yards'

    # Filter to only available features
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"\n📋 Using {len(available_features)} features:")
    for feat in available_features:
        print(f"  - {feat}")

    # STEP 6: Train/validation/test split (time-based)
    print(f"\n✂️  Splitting data (time-based)...")
    train_df, val_df, test_df = _time_based_split(df)

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    # Prepare X, y
    X_train = train_df[available_features].fillna(0)
    y_train = train_df[target_col]
    X_val = val_df[available_features].fillna(0)
    y_val = val_df[target_col]
    X_test = test_df[available_features].fillna(0)
    y_test = test_df[target_col]

    # STEP 7: Train model
    print(f"\n🤖 Training {model_type.upper()} model...")

    try:
        if model_type == 'xgboost':
            import xgboost as xgb

            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='reg:squarederror'
            )
        elif model_type == 'lightgbm':
            import lightgbm as lgb

            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train with sample weights
        train_weights = train_df['sample_weight'].values if 'sample_weight' in train_df.columns else None

        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        print(f"✓ Model trained successfully")

    except ImportError as e:
        print(f"⚠️  Model library not available: {e}")
        print(f"Install with: pip install {model_type}")
        return {
            'error': f'{model_type} not installed',
            'model_type': model_type,
            'season': season
        }

    # STEP 8: Evaluate on validation and test sets
    print(f"\n📈 Evaluating model performance...")

    # Validation metrics
    val_pred = model.predict(X_val)
    val_metrics = _calculate_metrics(y_val, val_pred)

    # Test metrics
    test_pred = model.predict(X_test)
    test_metrics = _calculate_metrics(y_test, test_pred)

    print(f"\n  Validation Set:")
    print(f"    RMSE:  {val_metrics['rmse']:.2f} yards")
    print(f"    MAE:   {val_metrics['mae']:.2f} yards")
    print(f"    R²:    {val_metrics['r_squared']:.3f}")

    print(f"\n  Test Set:")
    print(f"    RMSE:  {test_metrics['rmse']:.2f} yards")
    print(f"    MAE:   {test_metrics['mae']:.2f} yards")
    print(f"    R²:    {test_metrics['r_squared']:.3f}")

    # STEP 9: Feature importance
    print(f"\n🔍 Analyzing feature importance...")
    feature_importance = dict(zip(available_features, model.feature_importances_))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    print(f"\n  Top 10 features:")
    for i, (feat, importance) in enumerate(list(feature_importance.items())[:10], 1):
        print(f"    {i}. {feat}: {importance:.4f}")

    # STEP 10: Save model and metrics
    print(f"\n💾 Saving model to {output_model_path}")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save feature importance
    importance_file = output_model_path.parent / 'feature_importance.json'
    with open(importance_file, 'w') as f:
        json.dump(feature_importance, f, indent=2)

    print(f"✓ Model saved: {output_model_path}")
    print(f"✓ Feature importance saved: {importance_file}")

    # Return metrics
    metrics = {
        'model_type': model_type,
        'season': season,
        'num_train_samples': len(train_df),
        'num_val_samples': len(val_df),
        'num_test_samples': len(test_df),
        'num_features': len(available_features),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'top_features': list(feature_importance.items())[:10]
    }

    return metrics


def _prepare_training_data(player_features: Dict) -> pd.DataFrame:
    """Convert player features JSON to DataFrame.

    Args:
        player_features: Dict mapping player_id -> list of game features

    Returns:
        DataFrame with one row per player-game
    """
    rows = []

    for player_id, games in player_features.items():
        for game in games:
            # Determine if player is QB (has passing attempts)
            is_qb = 1 if game.get('attempts', 0) > 0 else 0

            row = {
                'player_id': player_id,
                'game_id': game.get('game_id', ''),
                'season': game.get('season', ''),
                'week': game.get('week', ''),
                'is_qb': is_qb,

                # Basic stats
                'passing_yards': game.get('passing_yards', 0),
                'passing_tds': game.get('passing_tds', 0),
                'completions': game.get('completions', 0),
                'attempts': game.get('attempts', 0),
                'interceptions': game.get('interceptions', 0),
                'sacks': game.get('sacks', 0),

                # Advanced metrics
                'total_epa': game.get('total_epa', 0),
                'qb_epa': game.get('qb_epa', 0),
                'cpoe_avg': game.get('cpoe_sum', 0) / max(game.get('cpoe_count', 1), 1),
                'success_plays': game.get('success_plays', 0),
                'total_plays': game.get('total_plays', 0),
                'wpa': game.get('wpa', 0),
                'air_epa': game.get('air_epa', 0),
                'yac_epa': game.get('yac_epa', 0),
                'xyac_avg': game.get('xyac_sum', 0) / max(game.get('xyac_count', 1), 1),
                'qb_hits': game.get('qb_hits', 0),
                'qb_hurries': game.get('qb_hurries', 0),
                'qb_pressures': game.get('qb_pressures', 0),
                'air_yards': game.get('air_yards', 0),
                'yards_after_catch': game.get('yards_after_catch', 0),
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    # Convert week to int (handle 'unknown' gracefully)
    df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)

    # Sort by player and week
    df = df.sort_values(['player_id', 'season', 'week'])

    return df


def _engineer_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer rolling window features.

    Args:
        df: DataFrame with player-game rows

    Returns:
        DataFrame with additional rolling features
    """
    # Group by player
    df = df.copy()

    # Calculate derived stats
    df['completion_pct'] = df['completions'] / df['attempts'].clip(lower=1) * 100
    df['yards_per_attempt'] = df['passing_yards'] / df['attempts'].clip(lower=1)
    df['success_rate'] = df['success_plays'] / df['total_plays'].clip(lower=1) * 100
    df['qb_pressure_rate'] = df['qb_pressures'] / df['attempts'].clip(lower=1) * 100
    df['qb_hit_rate'] = df['qb_hits'] / df['attempts'].clip(lower=1) * 100
    df['air_yards_avg'] = df['air_yards'] / df['completions'].clip(lower=1)

    # Games played counter
    df['games_played'] = df.groupby('player_id').cumcount()

    # Rolling averages (3-game, 5-game)
    for window in [3, 5]:
        df[f'qb_epa_roll_{window}'] = df.groupby('player_id')['qb_epa'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'total_epa_roll_{window}'] = df.groupby('player_id')['total_epa'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'cpoe_roll_{window}'] = df.groupby('player_id')['cpoe_avg'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'success_rate_roll_{window}'] = df.groupby('player_id')['success_rate'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'passing_yards_roll_{window}'] = df.groupby('player_id')['passing_yards'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Season averages (up to current week)
    df['qb_epa_avg'] = df.groupby('player_id')['qb_epa'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['total_epa_avg'] = df.groupby('player_id')['total_epa'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['wpa_avg'] = df.groupby('player_id')['wpa'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['air_epa_avg'] = df.groupby('player_id')['air_epa'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['yac_epa_avg'] = df.groupby('player_id')['yac_epa'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['passing_yards_avg'] = df.groupby('player_id')['passing_yards'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    return df


def _time_based_split(df: pd.DataFrame, train_pct: float = 0.7, val_pct: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by time (chronological).

    Args:
        df: DataFrame sorted by time
        train_pct: Percentage for training
        val_pct: Percentage for validation

    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def _calculate_recency_weights(seasons: pd.Series, decay: float = 0.3) -> pd.Series:
    """Calculate recency weights for seasons using exponential decay.

    More recent seasons get higher weight to reflect current NFL trends.

    Formula: weight = exp(-decay * years_ago)

    Example with decay=0.3:
        2025 (current): weight = 1.00
        2024 (1 year ago): weight = 0.74
        2023 (2 years ago): weight = 0.55
        2022 (3 years ago): weight = 0.41
        2021 (4 years ago): weight = 0.30
        2020 (5 years ago): weight = 0.22

    Args:
        seasons: Series of season years
        decay: Decay rate (higher = faster decay, typical range 0.2-0.5)

    Returns:
        Series of weights
    """
    current_season = seasons.max()
    years_ago = current_season - seasons
    weights = np.exp(-decay * years_ago)
    return weights


def _calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict:
    """Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dict with rmse, mae, r_squared
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)

    return {
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train passing model with advanced metrics (EPA, CPOE, success rate)'
    )
    parser.add_argument('--season', type=int, default=None,
                       help='Single season year (for backwards compatibility)')
    parser.add_argument('--seasons', type=int, nargs='+', default=None,
                       help='Multiple season years (e.g., --seasons 2020 2021 2022 2023 2024 2025)')
    parser.add_argument('--features', type=Path,
                       default=Path('outputs/player_pbp_features_by_id.json'),
                       help='Path to features file')
    parser.add_argument('--output', type=Path,
                       default=Path('outputs/models/passing_model.pkl'),
                       help='Path to save trained model')
    parser.add_argument('--model-type', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm'],
                       help='Model type to train')
    parser.add_argument('--no-recency-weighting', action='store_true',
                       help='Disable recency weighting (all seasons weighted equally)')
    parser.add_argument('--recency-decay', type=float, default=0.3,
                       help='Recency decay rate (default: 0.3, range: 0.2-0.5)')
    args = parser.parse_args()

    # Determine seasons to use
    if args.seasons:
        seasons = args.seasons
    elif args.season:
        seasons = [args.season]
    else:
        # Default: use all available seasons (2020-2025)
        seasons = [2020, 2021, 2022, 2023, 2024, 2025]
        print(f"No seasons specified, using default: {seasons}")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Train model
    metrics = train_passing_model(
        seasons=seasons,
        features_file=args.features,
        output_model_path=args.output,
        model_type=args.model_type,
        use_recency_weighting=not args.no_recency_weighting,
        recency_decay=args.recency_decay
    )

    # Print summary
    if 'error' not in metrics:
        print(f"\n{'='*60}")
        print(f"✓ Training Complete!")
        print(f"{'='*60}")
        print(f"\n📊 Final Test Metrics:")
        print(f"  RMSE:  {metrics['test_metrics']['rmse']:.2f} yards")
        print(f"  MAE:   {metrics['test_metrics']['mae']:.2f} yards")
        print(f"  R²:    {metrics['test_metrics']['r_squared']:.3f}")
        print(f"\n💾 Model saved to: {args.output}")
    else:
        print(f"\n⚠️  Training failed: {metrics.get('error')}")
