"""Train models for game derivative markets (1Q, 1H, 2H, Team Totals).

Markets trained:
- 1Q: Total, Spread, Moneyline
- 1H: Total, Spread, Moneyline (HIGH VOLUME)
- 2H: Total, Spread
- Team Totals: Home, Away

Uses same feature engineering as full-game models but targets quarter/half scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib


def load_games_with_quarters():
    """Load games data with quarter/half scores."""

    print(f"\n{'='*80}")
    print(f"LOADING GAMES WITH QUARTER SCORES")
    print(f"{'='*80}\n")

    games_file = Path('/home/user/nfl_backend/inputs/games_2025_with_quarters.csv')

    if not games_file.exists():
        print(f"❌ Games file not found: {games_file}")
        print("   Run: python -m backend.analysis.generate_quarter_scores")
        return pd.DataFrame()

    games = pd.read_csv(games_file)

    # Filter to 2025 regular season completed games
    games_2025 = games[
        (games['season'] == 2025) &
        (games['game_type'] == 'REG') &
        (games['home_score'].notna())
    ].copy()

    print(f"✅ Loaded {len(games_2025)} completed 2025 games")
    print()

    return games_2025


def create_game_features(games_df):
    """Create features for game derivative models."""

    df = games_df.copy()

    # Same features as full-game models
    df['rest_advantage'] = df['home_rest'] - df['away_rest']
    df['is_pickem'] = (df['spread_line'].abs() < 2.0).astype(int)
    df['is_high_total'] = (df['total_line'] > 48.0).astype(int)
    df['is_low_total'] = (df['total_line'] < 40.0).astype(int)
    df['home_implied_total'] = (df['total_line'] - df['spread_line']) / 2
    df['away_implied_total'] = (df['total_line'] + df['spread_line']) / 2
    df['is_dome'] = (df['roof'] == 'dome').astype(int)

    return df


def train_derivative_model(df, target_col, model_type='regression', training_weeks=None):
    """Train a derivative market model.

    Args:
        df: Games DataFrame with features
        target_col: Column to predict (e.g., 'h1_total', 'q1_spread')
        model_type: 'regression' or 'classification'
        training_weeks: List of weeks to train on

    Returns:
        Trained model, features list, metrics
    """

    if training_weeks is None:
        training_weeks = list(range(1, 10))

    # Filter to training weeks
    train_df = df[df['week'].isin(training_weeks)].copy()

    # Features
    feature_cols = [
        'spread_line',
        'total_line',
        'rest_advantage',
        'is_pickem',
        'is_high_total',
        'is_low_total',
        'div_game',
        'is_dome',
        'home_implied_total',
        'away_implied_total',
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df[target_col]

    print(f"  Training set: {len(X)} games")
    print(f"  Features: {len(feature_cols)} features")

    if model_type == 'regression':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)

        # Training MAE
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)

        print(f"  Training MAE: {mae:.2f}")

        return model, feature_cols, {'mae': mae}

    else:  # classification
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)

        # Training accuracy
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)

        print(f"  Training Accuracy: {acc:.1%}")

        return model, feature_cols, {'accuracy': acc}


def backtest_derivative_market(model, features, df, target_col, week, market_name):
    """Backtest a derivative market model.

    Args:
        model: Trained model
        features: List of feature columns
        df: Full games DataFrame
        target_col: Target column
        week: Week to backtest
        market_name: Name of market for display

    Returns:
        Dict with results
    """

    # Get week games
    week_df = df[df['week'] == week].copy()

    if len(week_df) == 0:
        return None

    X = week_df[features].fillna(0)
    y_actual = week_df[target_col]

    # Make predictions
    y_pred = model.predict(X)

    # Calculate MAE
    mae = mean_absolute_error(y_actual, y_pred)

    print(f"\n  Week {week}: {len(week_df)} games, MAE={mae:.2f}")

    # Show sample predictions
    for idx, (pred, actual) in enumerate(zip(y_pred[:5], y_actual[:5])):
        game = week_df.iloc[idx]
        away = game['away_team']
        home = game['home_team']

        diff = abs(pred - actual)
        status = '✅' if diff < 3 else '❌'

        print(f"    {status} {away} @ {home}: Pred={pred:.1f}, Actual={actual:.0f}, Error={diff:.1f}")

    return {
        'market': market_name,
        'week': week,
        'games': len(week_df),
        'mae': mae
    }


def main():
    """Train all game derivative market models."""

    print(f"\n{'='*80}")
    print(f"TRAINING GAME DERIVATIVE MARKET MODELS")
    print(f"{'='*80}\n")

    # Load data
    games_df = load_games_with_quarters()

    if len(games_df) == 0:
        return

    # Create features
    games_df = create_game_features(games_df)

    # Training configuration
    training_weeks = list(range(1, 10))
    test_weeks = [10, 11]

    print(f"Training weeks: {training_weeks}")
    print(f"Test weeks: {test_weeks}")
    print()

    # Models to train
    markets_config = {
        # 1Q Markets
        'q1_total': {
            'type': 'regression',
            'name': '1Q Total',
            'priority': 'MEDIUM',
        },
        'q1_spread': {
            'type': 'regression',
            'name': '1Q Spread',
            'priority': 'MEDIUM',
        },
        'q1_home_won': {
            'type': 'classification',
            'name': '1Q Moneyline',
            'priority': 'LOW',
        },

        # 1H Markets (HIGH PRIORITY)
        'h1_total': {
            'type': 'regression',
            'name': '1H Total',
            'priority': 'HIGH',
        },
        'h1_spread': {
            'type': 'regression',
            'name': '1H Spread',
            'priority': 'HIGH',
        },
        'h1_home_won': {
            'type': 'classification',
            'name': '1H Moneyline',
            'priority': 'HIGH',
        },

        # 2H Markets
        'h2_total': {
            'type': 'regression',
            'name': '2H Total',
            'priority': 'LOW',
        },
        'h2_spread': {
            'type': 'regression',
            'name': '2H Spread',
            'priority': 'LOW',
        },

        # Team Totals
        'home_total': {
            'type': 'regression',
            'name': 'Home Team Total',
            'priority': 'HIGH',
        },
        'away_total': {
            'type': 'regression',
            'name': 'Away Team Total',
            'priority': 'HIGH',
        },
    }

    # Train each market
    all_models = {}
    backtest_results = []

    for target_col, config in markets_config.items():
        print(f"\n{'#'*80}")
        print(f"# TRAINING: {config['name'].upper()} ({config['priority']} PRIORITY)")
        print(f"{'#'*80}\n")

        # Train model
        model, features, metrics = train_derivative_model(
            games_df,
            target_col,
            model_type=config['type'],
            training_weeks=training_weeks
        )

        # Save model
        models_dir = Path('outputs/models/derivative')
        models_dir.mkdir(parents=True, exist_ok=True)

        model_file = models_dir / f'{target_col}_model.pkl'
        joblib.dump({
            'model': model,
            'features': features,
            'config': config
        }, model_file)

        print(f"  💾 Saved to: {model_file}")

        all_models[target_col] = {
            'model': model,
            'features': features,
            'config': config
        }

        # Backtest on test weeks
        for week in test_weeks:
            result = backtest_derivative_market(
                model,
                features,
                games_df,
                target_col,
                week,
                config['name']
            )

            if result:
                result['priority'] = config['priority']
                backtest_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"TRAINING & BACKTESTING COMPLETE")
    print(f"{'='*80}\n")

    print(f"Markets trained: {len(all_models)}")
    print()

    # Backtest summary by priority
    if backtest_results:
        results_df = pd.DataFrame(backtest_results)

        for priority in ['HIGH', 'MEDIUM', 'LOW']:
            priority_results = results_df[results_df['priority'] == priority]

            if len(priority_results) > 0:
                print(f"\n{priority} PRIORITY MARKETS:")

                summary = priority_results.groupby('market').agg({
                    'games': 'sum',
                    'mae': 'mean'
                }).sort_values('mae')

                for market, row in summary.iterrows():
                    print(f"  {market:20s}: MAE {row['mae']:.2f} pts ({int(row['games'])} games)")

    print()
    print(f"\n✅ All derivative market models trained")
    print()

    print("New markets available:")
    print("  ✅ 1Q Total, 1Q Spread, 1Q Moneyline (3 markets)")
    print("  ✅ 1H Total, 1H Spread, 1H Moneyline (3 markets)")
    print("  ✅ 2H Total, 2H Spread (2 markets)")
    print("  ✅ Home Team Total, Away Team Total (2 markets)")
    print()
    print("Total new markets: 10")
    print()


if __name__ == '__main__':
    main()
