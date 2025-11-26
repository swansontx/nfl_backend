"""Train unified ML models for game outcomes (spreads, totals, moneylines).

Predicts home_score and away_score, from which we derive:
- Spread (margin) = home_score - away_score
- Total = home_score + away_score
- Win probability (for moneylines)

This unified approach ensures predictions are interconnected and consistent.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from dataclasses import dataclass
import joblib
import json

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error

from backend.backtesting.framework import BacktestingFramework, HistoricalGame


class UnifiedGameOutcomeMLPipeline:
    """ML pipeline for predicting game outcomes (spreads, totals, moneylines)."""

    def __init__(self, seasons: List[int] = [2021, 2022, 2023]):
        """Initialize pipeline.

        Args:
            seasons: Seasons to use for training/testing
        """
        self.framework = BacktestingFramework(seasons=seasons)
        self.feature_names = [
            'home_recent_ppg', 'away_recent_ppg',
            'home_season_ppg', 'away_season_ppg',
            'home_def_ppg_allowed', 'away_def_ppg_allowed',
            'home_off_vs_away_def', 'away_off_vs_home_def',
            'home_l3_ppg', 'away_l3_ppg',
            'home_l3_margin', 'away_l3_margin',
            'home_trend', 'away_trend',
            'home_std', 'away_std',
            'rest_differential', 'home_off_bye', 'away_off_bye',
            'temperature', 'wind_speed', 'is_cold', 'is_windy',
            'is_primetime', 'is_division_game', 'week'
        ]

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data.

        Returns:
            X_train, X_test, y_train, y_test
            where y is [home_score, away_score] for each game
        """
        print("Preparing training data...")

        all_data = []

        for season in self.framework.seasons:
            games = self.framework.load_historical_games(season)
            print(f"  Season {season}: {len(games)} games loaded")

            for game in games:
                # Skip early weeks (need history for features)
                if game.week < 5:
                    continue

                # Get historical games for both teams
                home_games = [g for g in games if g.week < game.week and
                            (g.home_team == game.home_team or g.away_team == game.home_team)]
                away_games = [g for g in games if g.week < game.week and
                            (g.home_team == game.away_team or g.away_team == game.away_team)]

                if len(home_games) < 3 or len(away_games) < 3:
                    continue

                # Extract scores for home team
                home_scores = []
                home_def_scores = []
                home_margins = []
                home_last_week = 0
                for g in home_games:
                    if g.home_team == game.home_team:
                        home_scores.append(g.home_score)
                        home_def_scores.append(g.away_score)
                        home_margins.append(g.home_score - g.away_score)
                        home_last_week = g.week
                    else:
                        home_scores.append(g.away_score)
                        home_def_scores.append(g.home_score)
                        home_margins.append(g.away_score - g.home_score)
                        home_last_week = g.week

                # Extract scores for away team
                away_scores = []
                away_def_scores = []
                away_margins = []
                away_last_week = 0
                for g in away_games:
                    if g.home_team == game.away_team:
                        away_scores.append(g.home_score)
                        away_def_scores.append(g.away_score)
                        away_margins.append(g.home_score - g.away_score)
                        away_last_week = g.week
                    else:
                        away_scores.append(g.away_score)
                        away_def_scores.append(g.home_score)
                        away_margins.append(g.away_score - g.home_score)
                        away_last_week = g.week

                # Calculate features
                home_recent_scores = home_scores[-4:]
                away_recent_scores = away_scores[-4:]
                home_l3_scores = home_scores[-3:]
                away_l3_scores = away_scores[-3:]
                home_l3_margins = home_margins[-3:]
                away_l3_margins = away_margins[-3:]
                home_all_scores = home_scores
                away_all_scores = away_scores

                # Rest differential
                rest_diff = 0  # Simplified

                row = {
                    # Recent form
                    'home_recent_ppg': np.mean(home_recent_scores),
                    'away_recent_ppg': np.mean(away_recent_scores),

                    # Season averages
                    'home_season_ppg': np.mean(home_all_scores),
                    'away_season_ppg': np.mean(away_all_scores),

                    # Defense
                    'home_def_ppg_allowed': np.mean(home_def_scores),
                    'away_def_ppg_allowed': np.mean(away_def_scores),

                    # Defense matchups
                    'home_off_vs_away_def': np.mean(home_all_scores) - np.mean(away_def_scores),
                    'away_off_vs_home_def': np.mean(away_all_scores) - np.mean(home_def_scores),

                    # Last 3
                    'home_l3_ppg': np.mean(home_l3_scores),
                    'away_l3_ppg': np.mean(away_l3_scores),
                    'home_l3_margin': np.mean(home_l3_margins),
                    'away_l3_margin': np.mean(away_l3_margins),

                    # Momentum
                    'home_trend': np.mean(home_recent_scores) - np.mean(home_all_scores),
                    'away_trend': np.mean(away_recent_scores) - np.mean(away_all_scores),

                    # Volatility
                    'home_std': np.std(home_recent_scores),
                    'away_std': np.std(away_recent_scores),

                    # Rest
                    'rest_differential': rest_diff,
                    'home_off_bye': 1 if (game.week - home_last_week) > 1 else 0,
                    'away_off_bye': 1 if (game.week - away_last_week) > 1 else 0,

                    # Weather
                    'temperature': game.temperature if game.temperature else 70.0,
                    'wind_speed': game.wind_speed if game.wind_speed else 0.0,
                    'is_cold': 1 if (game.temperature and game.temperature < 32) else 0,
                    'is_windy': 1 if (game.wind_speed and game.wind_speed > 15) else 0,

                    # Context
                    'is_primetime': 1 if game.is_primetime else 0,
                    'is_division_game': 1 if game.is_division_game else 0,
                    'week': game.week,

                    # Target
                    'home_score': game.home_score,
                    'away_score': game.away_score,
                    'season': season
                }

                all_data.append(row)

        df = pd.DataFrame(all_data)
        print(f"  Prepared {len(df)} training examples")

        # Handle missing values (fill with reasonable defaults)
        df = df.fillna({
            'temperature': 70.0,
            'wind_speed': 0.0,
            'is_cold': 0,
            'is_windy': 0,
            'is_primetime': 0,
            'is_division_game': 0,
            'rest_differential': 0,
            'home_off_bye': 0,
            'away_off_bye': 0
        })

        # Fill any remaining NaNs with column mean
        df = df.fillna(df.mean(numeric_only=True))

        # Split into train/test
        train_df = df[df['season'].isin([2021, 2022])]
        test_df = df[df['season'] == 2023]

        X_train = train_df[self.feature_names].values
        X_test = test_df[self.feature_names].values
        y_home_train = train_df['home_score'].values
        y_home_test = test_df['home_score'].values
        y_away_train = train_df['away_score'].values
        y_away_test = test_df['away_score'].values

        print(f"Train: {len(X_train)} games (2021-2022)")
        print(f"Test: {len(X_test)} games (2023)")
        print(f"Features: {len(self.feature_names)}")

        return X_train, X_test, (y_home_train, y_away_train), (y_home_test, y_away_test)

    def train_unified_model(
        self,
        X_train: np.ndarray,
        y_train: Tuple[np.ndarray, np.ndarray],
        X_test: np.ndarray,
        y_test: Tuple[np.ndarray, np.ndarray]
    ) -> Dict:
        """Train unified Neural Network that predicts both home and away scores.

        Args:
            X_train: Training features
            y_train: (home_scores, away_scores) for training
            X_test: Test features
            y_test: (home_scores, away_scores) for testing

        Returns:
            Model results dictionary
        """
        print("\n" + "="*80)
        print("UNIFIED NEURAL NETWORK (predicts home & away scores)")
        print("="*80)

        y_home_train, y_away_train = y_train
        y_home_test, y_away_test = y_test

        # Train separate models for home and away scores
        print("  Training home score predictor...")
        home_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            verbose=False
        )
        home_model.fit(X_train, y_home_train)

        print("  Training away score predictor...")
        away_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            verbose=False
        )
        away_model.fit(X_train, y_away_train)

        # Predictions
        home_pred_train = home_model.predict(X_train)
        away_pred_train = away_model.predict(X_train)
        home_pred_test = home_model.predict(X_test)
        away_pred_test = away_model.predict(X_test)

        # Calculate derived metrics
        # Spread = home - away
        spread_pred_train = home_pred_train - away_pred_train
        spread_actual_train = y_home_train - y_away_train
        spread_pred_test = home_pred_test - away_pred_test
        spread_actual_test = y_home_test - y_away_test

        # Total = home + away
        total_pred_train = home_pred_train + away_pred_train
        total_actual_train = y_home_train + y_away_train
        total_pred_test = home_pred_test + away_pred_test
        total_actual_test = y_home_test + y_away_test

        # Calculate MAEs
        spread_mae_train = mean_absolute_error(spread_actual_train, spread_pred_train)
        spread_mae_test = mean_absolute_error(spread_actual_test, spread_pred_test)
        total_mae_train = mean_absolute_error(total_actual_train, total_pred_train)
        total_mae_test = mean_absolute_error(total_actual_test, total_pred_test)

        # Baseline for spread: 0 (no advantage)
        spread_baseline = mean_absolute_error(spread_actual_test, np.zeros_like(spread_actual_test))

        # Baseline for total: mean of actuals
        total_baseline = mean_absolute_error(total_actual_test,
                                             np.full_like(total_actual_test, np.mean(total_actual_train)))

        print(f"\n  SPREAD Prediction:")
        print(f"    Train MAE: {spread_mae_train:.2f}")
        print(f"    Test MAE: {spread_mae_test:.2f}")
        print(f"    Baseline MAE: {spread_baseline:.2f}")
        improvement_spread = ((spread_baseline - spread_mae_test) / spread_baseline) * 100
        print(f"    Improvement: {improvement_spread:+.1f}%")

        print(f"\n  TOTAL Prediction:")
        print(f"    Train MAE: {total_mae_train:.2f}")
        print(f"    Test MAE: {total_mae_test:.2f}")
        print(f"    Baseline MAE: {total_baseline:.2f}")
        improvement_total = ((total_baseline - total_mae_test) / total_baseline) * 100
        print(f"    Improvement: {improvement_total:+.1f}%")

        # Overfitting
        spread_overfit = ((spread_mae_test - spread_mae_train) / spread_mae_train) * 100
        total_overfit = ((total_mae_test - total_mae_train) / total_mae_train) * 100
        print(f"\n  Overfitting (spread): {spread_overfit:.1f}%")
        print(f"  Overfitting (total): {total_overfit:.1f}%")

        # Save models
        models_dir = Path('models/unified_game_outcomes')
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(home_model, models_dir / 'home_score_nn.pkl')
        joblib.dump(away_model, models_dir / 'away_score_nn.pkl')
        print(f"\n  Saved to: {models_dir}")

        return {
            'home_model': home_model,
            'away_model': away_model,
            'spread_mae_test': spread_mae_test,
            'total_mae_test': total_mae_test,
            'spread_baseline': spread_baseline,
            'total_baseline': total_baseline,
            'spread_improvement': improvement_spread,
            'total_improvement': improvement_total
        }


def main():
    """Train unified game outcome models."""
    print("="*80)
    print("TRAINING UNIFIED GAME OUTCOME MODELS")
    print("="*80)
    print()

    # Initialize pipeline
    pipeline = UnifiedGameOutcomeMLPipeline(seasons=[2021, 2022, 2023])

    # Prepare data
    X_train, X_test, y_train, y_test = pipeline.prepare_data()

    # Baseline metrics
    y_home_test, y_away_test = y_test
    spread_baseline = mean_absolute_error(
        y_home_test - y_away_test,
        np.zeros(len(y_home_test))
    )
    total_baseline = mean_absolute_error(
        y_home_test + y_away_test,
        np.full(len(y_home_test), np.mean(y_train[0] + y_train[1]))
    )

    print(f"\nBaseline Spread MAE: {spread_baseline:.2f}")
    print(f"Baseline Total MAE: {total_baseline:.2f}")
    print()

    # Train unified model
    results = pipeline.train_unified_model(X_train, y_train, X_test, y_test)

    # Save metadata
    models_dir = Path('models/unified_game_outcomes')
    metadata = {
        'model_type': 'unified_neural_network',
        'features': pipeline.feature_names,
        'spread_mae': float(results['spread_mae_test']),
        'total_mae': float(results['total_mae_test']),
        'spread_baseline': float(results['spread_baseline']),
        'total_baseline': float(results['total_baseline']),
        'spread_improvement': float(results['spread_improvement']),
        'total_improvement': float(results['total_improvement']),
        'architecture': 'MLPRegressor (100, 50, 25)',
        'trained_on': '2021-2023 seasons',
        'feature_count': len(pipeline.feature_names)
    }

    with open(models_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\n✅ Unified Neural Network trained!")
    print(f"\n  SPREADS:")
    print(f"    Test MAE: {results['spread_mae_test']:.2f}")
    print(f"    Baseline: {results['spread_baseline']:.2f}")
    print(f"    Improvement: {results['spread_improvement']:+.1f}%")
    print(f"\n  TOTALS:")
    print(f"    Test MAE: {results['total_mae_test']:.2f}")
    print(f"    Baseline: {results['total_baseline']:.2f}")
    print(f"    Improvement: {results['total_improvement']:+.1f}%")
    print(f"\n  Models saved: models/unified_game_outcomes/")
    print()


if __name__ == '__main__':
    main()
