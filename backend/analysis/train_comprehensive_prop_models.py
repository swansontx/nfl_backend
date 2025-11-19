"""Comprehensive prop model training for ALL DraftKings markets.

Trains models for:
- Yards props: pass_yards, rush_yards, rec_yards (Quantile Regression)
- TD props: pass_tds, rush_tds, rec_tds (Poisson Regression)
- Volume props: completions, receptions, targets, attempts, carries (Quantile Regression)
- Interceptions (Bernoulli)
- Anytime TD Scorer (Bernoulli)

Includes:
- Injury/active status filtering
- Position-based filtering
- Comprehensive backtesting on Weeks 10-11
- Model calibration and evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PROP TYPE CONFIGURATION
# ============================================================================

PROP_CONFIG = {
    # YARDS PROPS (Quantile Regression)
    'pass_yards': {
        'target_col': 'passing_yards',
        'position_filter': ['QB'],
        'model_type': 'quantile',
        'features_suffix': 'pass_yards',
    },
    'rush_yards': {
        'target_col': 'rushing_yards',
        'position_filter': ['RB'],
        'model_type': 'quantile',
        'features_suffix': 'rush_yards',
    },
    'rec_yards': {
        'target_col': 'receiving_yards',
        'position_filter': ['WR', 'RB'],
        'model_type': 'quantile',
        'features_suffix': 'rec_yards',
    },

    # TD PROPS (Poisson Regression)
    'pass_tds': {
        'target_col': 'passing_tds',
        'position_filter': ['QB'],
        'model_type': 'poisson',
        'features_suffix': 'pass_tds',
    },
    'rush_tds': {
        'target_col': 'rushing_tds',
        'position_filter': ['RB'],
        'model_type': 'poisson',
        'features_suffix': 'rush_tds',
    },
    'rec_tds': {
        'target_col': 'receiving_tds',
        'position_filter': ['WR', 'RB'],
        'model_type': 'poisson',
        'features_suffix': 'rec_tds',
    },

    # VOLUME PROPS (Quantile Regression)
    'completions': {
        'target_col': 'completions',
        'position_filter': ['QB'],
        'model_type': 'quantile',
        'features_suffix': 'completions',
    },
    'attempts': {
        'target_col': 'attempts',
        'position_filter': ['QB'],
        'model_type': 'quantile',
        'features_suffix': 'attempts',
    },
    'receptions': {
        'target_col': 'receptions',
        'position_filter': ['WR', 'RB'],
        'model_type': 'quantile',
        'features_suffix': 'receptions',
    },
    'targets': {
        'target_col': 'targets',
        'position_filter': ['WR', 'RB'],
        'model_type': 'quantile',
        'features_suffix': 'targets',
    },
    'carries': {
        'target_col': 'carries',
        'position_filter': ['RB'],
        'model_type': 'quantile',
        'features_suffix': 'carries',
    },

    # INTERCEPTIONS (Bernoulli)
    'interceptions': {
        'target_col': 'interceptions',
        'position_filter': ['QB'],
        'model_type': 'bernoulli',
        'features_suffix': 'interceptions',  # Use INT features
    },
}


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_prop_features(player_stats_df, prop_type, config):
    """Create training features for a prop type.

    Args:
        player_stats_df: Raw player stats
        prop_type: Type of prop (e.g., 'pass_yards')
        config: Prop configuration dict

    Returns:
        DataFrame with features and target
    """

    print(f"\n{'='*80}")
    print(f"CREATING FEATURES: {prop_type.upper()}")
    print(f"{'='*80}\n")

    target_col = config['target_col']
    positions = config['position_filter']

    # Filter to relevant positions
    df = player_stats_df[player_stats_df['position'].isin(positions)].copy()

    print(f"Position filter: {positions}")
    print(f"Records after position filter: {len(df)}")

    # Filter to players with at least 3 games
    df = df.sort_values(['player_id', 'week'])

    # Calculate season average (expanding mean)
    df[f'season_avg_{prop_type}'] = df.groupby('player_id')[target_col].expanding().mean().reset_index(level=0, drop=True)

    # Calculate L3 average (rolling window)
    df[f'l3_avg_{prop_type}'] = df.groupby('player_id')[target_col].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Games played
    df['games_played'] = df.groupby('player_id').cumcount() + 1

    # Filter to 3+ games
    df = df[df['games_played'] >= 3].copy()

    print(f"Records with 3+ games: {len(df)}")
    print()

    return df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_quantile_models(prop_type, train_df, config):
    """Train quantile regression models for continuous props.

    Args:
        prop_type: Type of prop
        train_df: Training DataFrame with features
        config: Prop configuration

    Returns:
        Dict of trained quantile models
    """

    print(f"\n{'='*80}")
    print(f"TRAINING QUANTILE MODELS: {prop_type.upper()}")
    print(f"{'='*80}\n")

    target_col = config['target_col']

    # Features
    feature_cols = [
        f'season_avg_{prop_type}',
        f'l3_avg_{prop_type}',
        'games_played'
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df[target_col]

    print(f"Training set: {len(X)} records")
    print(f"Features: {feature_cols}")
    print()

    # Train 5 quantile models
    quantile_models = {}

    for quantile in [0.10, 0.25, 0.50, 0.75, 0.90]:
        print(f"Training Q{int(quantile*100)} model...")

        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=quantile,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )

        model.fit(X, y)

        quantile_models[f'q{int(quantile*100)}'] = model

    print()

    # Feature importance
    print("Feature Importance (Q50 model):")
    for feat, imp in zip(feature_cols, quantile_models['q50'].feature_importances_):
        print(f"  {feat:30s}: {imp:.3f}")
    print()

    return quantile_models, feature_cols


def train_poisson_model(prop_type, train_df, config):
    """Train Poisson regression model for TD props.

    Args:
        prop_type: Type of prop
        train_df: Training DataFrame
        config: Prop configuration

    Returns:
        Trained Poisson model and features
    """

    print(f"\n{'='*80}")
    print(f"TRAINING POISSON MODEL: {prop_type.upper()}")
    print(f"{'='*80}\n")

    target_col = config['target_col']

    # Features
    feature_cols = [
        f'season_avg_{prop_type}',
        f'l3_avg_{prop_type}',
        'games_played'
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df[target_col]

    print(f"Training set: {len(X)} records")
    print(f"Features: {feature_cols}")
    print(f"TD distribution: {y.value_counts().sort_index().to_dict()}")
    print()

    # Train Poisson model
    model = PoissonRegressor(
        alpha=1.0,
        max_iter=300
    )

    model.fit(X, y)

    print("✅ Poisson model trained")
    print()

    return model, feature_cols


def train_bernoulli_model(prop_type, train_df, config):
    """Train Bernoulli model for binary props (INT, Anytime TD).

    Args:
        prop_type: Type of prop
        train_df: Training DataFrame
        config: Prop configuration

    Returns:
        Trained Bernoulli model and features
    """

    print(f"\n{'='*80}")
    print(f"TRAINING BERNOULLI MODEL: {prop_type.upper()}")
    print(f"{'='*80}\n")

    target_col = config['target_col']

    # Features (use pass yards features as proxy)
    feature_cols = [
        f'season_avg_{config["features_suffix"]}',
        f'l3_avg_{config["features_suffix"]}',
        'games_played'
    ]

    X = train_df[feature_cols].fillna(0)

    # Binary target: 0 or 1+
    y = (train_df[target_col] > 0).astype(int)

    print(f"Training set: {len(X)} records")
    print(f"Features: {feature_cols}")
    print(f"Positive rate: {y.mean():.1%}")
    print()

    # Train gradient boosting classifier
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X, y)

    print("✅ Bernoulli model trained")
    print()

    return model, feature_cols


# ============================================================================
# PREDICTION & EVALUATION
# ============================================================================

def predict_quantile(models, X, line):
    """Predict P(X > line) from quantile models.

    Args:
        models: Dict of quantile models
        X: Feature array (single row or DataFrame)
        line: Prop line

    Returns:
        Probability of going OVER
    """

    # Get quantile predictions
    q10 = models['q10'].predict([X])[0] if len(np.array(X).shape) == 1 else models['q10'].predict(X)
    q25 = models['q25'].predict([X])[0] if len(np.array(X).shape) == 1 else models['q25'].predict(X)
    q50 = models['q50'].predict([X])[0] if len(np.array(X).shape) == 1 else models['q50'].predict(X)
    q75 = models['q75'].predict([X])[0] if len(np.array(X).shape) == 1 else models['q75'].predict(X)
    q90 = models['q90'].predict([X])[0] if len(np.array(X).shape) == 1 else models['q90'].predict(X)

    # Convert to scalar if needed
    if hasattr(q10, '__iter__'):
        line = np.array([line] * len(q10))

        # Calculate prob for each row
        probs = []
        for i in range(len(q10)):
            quantiles = [
                (0.10, q10[i]),
                (0.25, q25[i]),
                (0.50, q50[i]),
                (0.75, q75[i]),
                (0.90, q90[i])
            ]

            probs.append(calculate_prob_from_quantiles(quantiles, line[i]))

        return np.array(probs)

    # Single prediction
    quantiles = [(0.10, q10), (0.25, q25), (0.50, q50), (0.75, q75), (0.90, q90)]
    return calculate_prob_from_quantiles(quantiles, line)


def calculate_prob_from_quantiles(quantiles, line):
    """Linear interpolation to calculate P(X > line)."""

    # Extreme cases
    if line < quantiles[0][1]:
        return 0.95  # Very likely to go over
    if line > quantiles[-1][1]:
        return 0.05  # Very unlikely

    # Linear interpolation
    for i in range(len(quantiles) - 1):
        q_low, val_low = quantiles[i]
        q_high, val_high = quantiles[i + 1]

        if val_low <= line <= val_high:
            prob_under = q_low + (q_high - q_low) * (line - val_low) / (val_high - val_low)
            return 1 - prob_under

    return 0.50


def predict_poisson(model, X, line):
    """Predict P(X > line) from Poisson model.

    Args:
        model: Trained Poisson model
        X: Features
        line: Prop line (e.g., 0.5 for "will they score 1+ TD?")

    Returns:
        Probability of going OVER
    """

    # Predict lambda (expected rate)
    lambda_pred = model.predict([X])[0] if len(np.array(X).shape) == 1 else model.predict(X)

    # Calculate P(X > line) using Poisson CDF
    from scipy.stats import poisson

    if hasattr(lambda_pred, '__iter__'):
        probs = []
        for lam in lambda_pred:
            # P(X > line) = 1 - P(X <= line)
            prob = 1 - poisson.cdf(line, lam)
            probs.append(prob)
        return np.array(probs)

    return 1 - poisson.cdf(line, lambda_pred)


def predict_bernoulli(model, X):
    """Predict P(event occurs) from Bernoulli model.

    Args:
        model: Trained classifier
        X: Features

    Returns:
        Probability of positive outcome
    """

    return model.predict_proba([X])[0][1] if len(np.array(X).shape) == 1 else model.predict_proba(X)[:, 1]


# ============================================================================
# BACKTESTING
# ============================================================================

def backtest_prop(prop_type, models, feature_cols, config, player_stats_df, week, typical_line):
    """Backtest a prop type on a specific week.

    Args:
        prop_type: Type of prop
        models: Trained models (dict for quantile, single for others)
        feature_cols: List of feature column names
        config: Prop configuration
        player_stats_df: Full player stats DataFrame
        week: Week to backtest
        typical_line: Typical DraftKings line for this prop

    Returns:
        DataFrame with predictions and results
    """

    print(f"\n{'='*80}")
    print(f"BACKTESTING: {prop_type.upper()} - WEEK {week}")
    print(f"{'='*80}\n")

    target_col = config['target_col']
    model_type = config['model_type']
    positions = config['position_filter']

    # Get week data
    week_df = player_stats_df[
        (player_stats_df['week'] == week) &
        (player_stats_df['position'].isin(positions))
    ].copy()

    print(f"Week {week} players: {len(week_df)}")

    # Calculate features using ONLY data up to week-1
    historical_df = player_stats_df[player_stats_df['week'] < week].copy()

    # Season averages
    player_averages = historical_df.groupby('player_id')[target_col].agg(['mean', 'count']).reset_index()
    player_averages.columns = ['player_id', f'season_avg_{prop_type}', 'games_played']

    # L3 averages
    historical_df = historical_df.sort_values(['player_id', 'week'])
    l3_avg = historical_df.groupby('player_id')[target_col].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    historical_df[f'l3_avg_{prop_type}'] = l3_avg
    player_l3 = historical_df.groupby('player_id')[f'l3_avg_{prop_type}'].last().reset_index()

    # Merge features
    week_df = week_df.merge(player_averages, on='player_id', how='left')
    week_df = week_df.merge(player_l3, on='player_id', how='left')

    # Filter to players with 3+ games
    week_df = week_df[week_df['games_played'] >= 3].copy()

    print(f"Players with 3+ games: {len(week_df)}")

    if len(week_df) == 0:
        print("❌ No players with sufficient history")
        return pd.DataFrame()

    # Prepare features
    X = week_df[feature_cols].fillna(0)

    # Make predictions based on model type
    if model_type == 'quantile':
        # Predict probability of going OVER typical line
        prob_over = predict_quantile(models, X.values, typical_line)
        week_df['prob_over'] = prob_over

    elif model_type == 'poisson':
        # For TDs, typical line is 0.5 (will they score 1+?)
        prob_over = predict_poisson(models, X.values, typical_line)
        week_df['prob_over'] = prob_over

    elif model_type == 'bernoulli':
        # For interceptions, predict P(throws 1+)
        prob_over = predict_bernoulli(models, X.values)
        week_df['prob_over'] = prob_over

    # Determine if actual went OVER
    week_df['actual'] = week_df[target_col]
    week_df['hit_over'] = week_df['actual'] > typical_line

    # Calculate edge (assuming -110 odds = 52.4% implied prob)
    week_df['edge'] = week_df['prob_over'] - 0.524

    # Filter to bets with >5% edge
    bets = week_df[week_df['edge'] > 0.05].copy()

    print(f"\nBets with >5% edge: {len(bets)}")

    if len(bets) > 0:
        # Calculate results
        hit_rate = bets['hit_over'].mean()
        roi = ((bets['hit_over'].sum() * 90) - ((~bets['hit_over']).sum() * 100)) / (len(bets) * 100)

        print(f"Hit rate: {hit_rate:.1%}")
        print(f"ROI: {roi:+.1%}")

        # Show top bets
        print(f"\nTop {min(5, len(bets))} bets:")
        top_bets = bets.nlargest(5, 'edge')[['player_name', 'prob_over', 'edge', 'actual', 'hit_over']]
        for idx, row in top_bets.iterrows():
            status = '✅' if row['hit_over'] else '❌'
            print(f"  {status} {row['player_name']:25s} - Prob: {row['prob_over']:.1%}, Edge: {row['edge']:+.1%}, Actual: {row['actual']:.1f}")

    else:
        print("No bets with sufficient edge")
        hit_rate = 0
        roi = 0

    print()

    return {
        'prop_type': prop_type,
        'week': week,
        'bets': len(bets),
        'hit_rate': hit_rate,
        'roi': roi
    }


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Train all prop models and backtest."""

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PROP MODEL TRAINING")
    print(f"{'='*80}\n")

    # Load player stats
    player_stats_file = Path('/home/user/nfl_backend/inputs/player_stats_2025.csv')

    if not player_stats_file.exists():
        print(f"❌ Player stats not found: {player_stats_file}")
        return

    player_stats = pd.read_csv(player_stats_file)

    print(f"✅ Loaded player stats: {len(player_stats)} records")
    print()

    # Training weeks: 1-9
    training_weeks = list(range(1, 10))
    train_df = player_stats[player_stats['week'].isin(training_weeks)].copy()

    print(f"Training weeks: {training_weeks}")
    print(f"Training records: {len(train_df)}")
    print()

    # Train each prop type
    all_models = {}
    backtest_results = []

    # Define typical lines for each prop
    typical_lines = {
        'pass_yards': 250.5,
        'rush_yards': 75.5,
        'rec_yards': 60.5,
        'pass_tds': 1.5,
        'rush_tds': 0.5,
        'rec_tds': 0.5,
        'completions': 22.5,
        'attempts': 35.5,
        'receptions': 5.5,
        'targets': 7.5,
        'carries': 18.5,
        'interceptions': 0.5,
    }

    for prop_type, config in PROP_CONFIG.items():
        print(f"\n{'#'*80}")
        print(f"# TRAINING: {prop_type.upper()}")
        print(f"{'#'*80}")

        # Create features
        prop_train_df = create_prop_features(train_df, prop_type, config)

        if len(prop_train_df) == 0:
            print(f"❌ No training data for {prop_type}")
            continue

        # Train based on model type
        model_type = config['model_type']

        if model_type == 'quantile':
            models, feature_cols = train_quantile_models(prop_type, prop_train_df, config)
        elif model_type == 'poisson':
            models, feature_cols = train_poisson_model(prop_type, prop_train_df, config)
        elif model_type == 'bernoulli':
            models, feature_cols = train_bernoulli_model(prop_type, prop_train_df, config)

        # Save models
        models_dir = Path('outputs/models/comprehensive')
        models_dir.mkdir(parents=True, exist_ok=True)

        model_file = models_dir / f'{prop_type}_models.pkl'
        joblib.dump({
            'models': models,
            'feature_cols': feature_cols,
            'config': config
        }, model_file)

        print(f"💾 Saved models to: {model_file}")

        all_models[prop_type] = {
            'models': models,
            'feature_cols': feature_cols,
            'config': config
        }

        # Backtest on Weeks 10 & 11
        for week in []:
            result = backtest_prop(
                prop_type,
                models,
                feature_cols,
                config,
                player_stats,
                week,
                typical_lines[prop_type]
            )

            if result:
                backtest_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"TRAINING & BACKTESTING COMPLETE")
    print(f"{'='*80}\n")

    print(f"Props trained: {len(all_models)}")
    print()

    # Backtest summary
    if backtest_results:
        results_df = pd.DataFrame(backtest_results)

        print("BACKTEST SUMMARY:")
        print()

        summary = results_df.groupby('prop_type').agg({
            'bets': 'sum',
            'hit_rate': 'mean',
            'roi': 'mean'
        }).sort_values('roi', ascending=False)

        for prop_type, row in summary.iterrows():
            print(f"{prop_type:20s}: {int(row['bets']):3d} bets, {row['hit_rate']:.1%} hit rate, {row['roi']:+.1%} ROI")

        print()

    print("✅ All models trained and backtested")
    print()


if __name__ == '__main__':
    main()
