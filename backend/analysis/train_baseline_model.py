"""Train baseline model on 2025 Weeks 1-9 data.

Simple gradient boosting model using game-level features.
Test on Week 10 to see real performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss
import json


def load_training_data():
    """Load processed training data."""

    print(f"\n{'='*80}")
    print(f"TRAINING BASELINE MODEL")
    print(f"{'='*80}\n")

    train_file = Path('outputs/training_data_weeks_1_9.csv')

    if not train_file.exists():
        print(f"❌ Training data not found: {train_file}")
        print(f"   Run: python -m backend.analysis.fetch_2025_training_data")
        return pd.DataFrame()

    train_df = pd.read_csv(train_file)

    print(f"✅ Loaded training data: {len(train_df)} games")
    print()

    return train_df


def prepare_features(df):
    """Prepare feature matrix for modeling."""

    print("🔧 Preparing features...")

    # Select features
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

    # Filter to features that exist
    available_features = [f for f in feature_cols if f in df.columns]

    print(f"   Features used: {available_features}")
    print()

    X = df[available_features].copy()

    # Fill any missing values
    X = X.fillna(0)

    return X, available_features


def train_models(train_df):
    """Train models for total score, spread, and winner."""

    X, features = prepare_features(train_df)

    models = {}

    # Target 1: Total score
    print("📈 Training TOTAL SCORE model...")
    y_total = train_df['actual_total']

    total_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    total_model.fit(X, y_total)

    # Evaluate on training set
    y_pred = total_model.predict(X)
    mae = mean_absolute_error(y_total, y_pred)

    print(f"   Training MAE: {mae:.2f} points")
    print()

    models['total'] = total_model

    # Target 2: Spread
    print("📊 Training SPREAD model...")
    y_spread = train_df['actual_spread']

    spread_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    spread_model.fit(X, y_spread)

    y_pred = spread_model.predict(X)
    mae = mean_absolute_error(y_spread, y_pred)

    print(f"   Training MAE: {mae:.2f} points")
    print()

    models['spread'] = spread_model

    # Target 3: Home winner
    print("🎯 Training HOME WINNER model...")
    y_winner = train_df['home_won']

    winner_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    winner_model.fit(X, y_winner)

    y_pred = winner_model.predict(X)
    accuracy = accuracy_score(y_winner, y_pred)

    y_pred_proba = winner_model.predict_proba(X)[:, 1]
    logloss = log_loss(y_winner, y_pred_proba)

    print(f"   Training Accuracy: {accuracy:.1%}")
    print(f"   Training Log Loss: {logloss:.4f}")
    print()

    models['winner'] = winner_model

    # Feature importance
    print("💡 Top Features (Total Score Model):")
    importance = pd.DataFrame({
        'feature': features,
        'importance': total_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in importance.head(5).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    print()

    return models, features


def make_week10_predictions(models, features):
    """Make predictions for Week 10."""

    print(f"\n{'='*80}")
    print(f"MAKING WEEK 10 PREDICTIONS")
    print(f"{'='*80}\n")

    # Load games data
    games_file = Path('/home/user/nfl_backend/inputs/games_2025.csv')
    all_games = pd.read_csv(games_file)

    week10 = all_games[
        (all_games['season'] == 2025) &
        (all_games['week'] == 10) &
        (all_games['game_type'] == 'REG')
    ].copy()

    print(f"📅 Week 10 games: {len(week10)}")
    print()

    # Create features for Week 10
    week10['rest_advantage'] = week10['home_rest'] - week10['away_rest']
    week10['is_pickem'] = (week10['spread_line'].abs() < 2.0).astype(int)
    week10['is_high_total'] = (week10['total_line'] > 48.0).astype(int)
    week10['is_low_total'] = (week10['total_line'] < 40.0).astype(int)
    week10['home_implied_total'] = (week10['total_line'] - week10['spread_line']) / 2
    week10['away_implied_total'] = (week10['total_line'] + week10['spread_line']) / 2
    week10['is_dome'] = (week10['roof'] == 'dome').astype(int)

    X_test, _ = prepare_features(week10)

    # Make predictions
    week10['predicted_total'] = models['total'].predict(X_test)
    week10['predicted_spread'] = models['spread'].predict(X_test)
    week10['home_win_prob'] = models['winner'].predict_proba(X_test)[:, 1]

    # Calculate predicted scores
    week10['predicted_home_score'] = (week10['predicted_total'] - week10['predicted_spread']) / 2
    week10['predicted_away_score'] = (week10['predicted_total'] + week10['predicted_spread']) / 2

    # Determine predicted winner
    week10['predicted_winner'] = week10.apply(
        lambda x: x['home_team'] if x['home_win_prob'] > 0.5 else x['away_team'],
        axis=1
    )

    print("🎲 Model Predictions:")
    print()

    for idx, game in week10.iterrows():
        away = game['away_team']
        home = game['home_team']

        pred_total = game['predicted_total']
        pred_spread = game['predicted_spread']
        home_prob = game['home_win_prob']

        print(f"  {away} @ {home}")
        print(f"    Predicted Total: {pred_total:.1f} (line: {game['total_line']:.1f})")
        print(f"    Predicted Spread: {pred_spread:+.1f} (line: {game['spread_line']:+.1f})")
        print(f"    Home Win Prob: {home_prob:.1%}")

        # Identify value bets
        if abs(pred_total - game['total_line']) > 3:
            side = "OVER" if pred_total > game['total_line'] else "UNDER"
            edge = abs(pred_total - game['total_line'])
            print(f"    💰 VALUE: {side} total (edge: {edge:.1f} pts)")

        print()

    return week10


def compare_to_actuals(predictions):
    """Compare predictions to actual Week 10 results."""

    print(f"\n{'='*80}")
    print(f"COMPARING TO ACTUAL RESULTS")
    print(f"{'='*80}\n")

    # Calculate actual outcomes
    predictions['actual_total'] = predictions['home_score'] + predictions['away_score']
    predictions['actual_spread'] = predictions['home_score'] - predictions['away_score']
    predictions['actual_winner'] = predictions.apply(
        lambda x: x['home_team'] if x['home_score'] > x['away_score'] else x['away_team'],
        axis=1
    )

    # Remove incomplete games
    completed = predictions[predictions['actual_total'].notna()].copy()

    print(f"📊 Completed games: {len(completed)}")
    print()

    # Evaluate total predictions
    total_mae = mean_absolute_error(completed['actual_total'], completed['predicted_total'])
    print(f"🎯 Total Score MAE: {total_mae:.2f} points")

    # Evaluate spread predictions
    spread_mae = mean_absolute_error(completed['actual_spread'], completed['predicted_spread'])
    print(f"🎯 Spread MAE: {spread_mae:.2f} points")

    # Evaluate winner predictions
    winner_acc = (completed['predicted_winner'] == completed['actual_winner']).mean()
    print(f"🎯 Winner Accuracy: {winner_acc:.1%}")
    print()

    # Game-by-game
    print("📋 GAME-BY-GAME RESULTS:")
    print()

    for idx, game in completed.iterrows():
        away = game['away_team']
        home = game['home_team']

        pred_total = game['predicted_total']
        actual_total = game['actual_total']
        total_error = abs(pred_total - actual_total)

        pred_winner = game['predicted_winner']
        actual_winner = game['actual_winner']
        winner_status = "✅" if pred_winner == actual_winner else "❌"

        print(f"  {away} @ {home}")
        print(f"    Predicted: {game['predicted_away_score']:.0f}-{game['predicted_home_score']:.0f}")
        print(f"    Actual: {game['away_score']:.0f}-{game['home_score']:.0f}")
        print(f"    Total: pred {pred_total:.0f}, actual {actual_total:.0f} (error: {total_error:.1f})")
        print(f"    Winner: {winner_status} ({pred_winner})")
        print()

    return {
        'total_mae': total_mae,
        'spread_mae': spread_mae,
        'winner_accuracy': winner_acc,
        'games': len(completed),
    }


def save_results(predictions, metrics):
    """Save predictions and metrics."""

    outputs_dir = Path('outputs')
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pred_file = outputs_dir / 'week10_predictions.csv'
    predictions.to_csv(pred_file, index=False)

    print(f"💾 Saved predictions to: {pred_file}")

    # Save metrics
    metrics_file = outputs_dir / 'week10_metrics.json'

    with open(metrics_file, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in metrics.items()}, f, indent=2)

    print(f"💾 Saved metrics to: {metrics_file}")
    print()


def main():
    """Main training and evaluation pipeline."""

    # 1. Load training data
    train_df = load_training_data()

    if len(train_df) == 0:
        return

    # 2. Train models
    models, features = train_models(train_df)

    # 3. Make Week 10 predictions
    predictions = make_week10_predictions(models, features)

    # 4. Compare to actuals
    metrics = compare_to_actuals(predictions)

    # 5. Save results
    save_results(predictions, metrics)

    print(f"\n{'='*80}")
    print(f"✅ BASELINE MODEL COMPLETE")
    print(f"{'='*80}\n")

    print(f"📊 Summary:")
    print(f"   Total MAE: {metrics['total_mae']:.2f} points")
    print(f"   Spread MAE: {metrics['spread_mae']:.2f} points")
    print(f"   Winner Accuracy: {metrics['winner_accuracy']:.1%}")
    print()

    print(f"💡 Comparison to Vegas lines:")
    print(f"   Our total MAE: {metrics['total_mae']:.2f}")
    print(f"   Vegas total MAE: 11.67 (from earlier backtest)")
    print(f"   Improvement: {11.67 - metrics['total_mae']:+.2f} points")
    print()


if __name__ == '__main__':
    main()
