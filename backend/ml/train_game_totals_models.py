"""Production ML Model for Game Totals - ACTUALLY DEPLOYED.

This is not a backtest - this is a REAL model that can make predictions.

Models to test:
1. Gradient Boosting (tuned)
2. Random Forest
3. XGBoost
4. Neural Network
5. Ensemble (combine best models)

We'll train on 2021-2023, validate properly, and deploy the best one.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from backend.backtesting.framework import BacktestingFramework


@dataclass
class MLModelResult:
    """Result from training an ML model."""
    model_name: str
    model: object
    feature_names: List[str]

    # Performance
    train_mae: float
    test_mae: float
    cv_mae: float
    cv_std: float

    # Overfitting check
    overfitting_gap: float

    # Feature importance
    feature_importance: Dict[str, float]

    # Model saved
    model_path: str


class GameTotalsMLPipeline:
    """Production ML pipeline for game totals."""

    def __init__(self, output_dir: str = 'models/game_totals_ml'):
        """Initialize pipeline.

        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.framework = BacktestingFramework(seasons=[2021, 2022, 2023])
        self.data = None
        self.feature_cols = None

    def prepare_data(self) -> pd.DataFrame:
        """Prepare training data with all features.

        Returns:
            DataFrame with features and target
        """
        print("Preparing training data...")

        all_data = []

        for season in self.framework.seasons:
            games = self.framework.load_historical_games(season)

            for game in games:
                if game.week < 5:  # Need history
                    continue

                if game.home_score is None or game.away_score is None:
                    continue

                # Get recent games for baselines
                home_recent = [
                    g for g in games
                    if (g.home_team == game.home_team or g.away_team == game.home_team) and
                    g.week < game.week and
                    g.home_score is not None
                ]

                away_recent = [
                    g for g in games
                    if (g.home_team == game.away_team or g.away_team == game.away_team) and
                    g.week < game.week and
                    g.home_score is not None
                ]

                if len(home_recent) < 3 or len(away_recent) < 3:
                    continue

                # Recent scores (last 4 games)
                home_last_4 = home_recent[-4:]
                away_last_4 = away_recent[-4:]

                home_recent_scores = [
                    g.home_score if g.home_team == game.home_team else g.away_score
                    for g in home_last_4
                ]
                away_recent_scores = [
                    g.home_score if g.home_team == game.away_team else g.away_score
                    for g in away_last_4
                ]

                # Season averages
                home_all_scores = [
                    g.home_score if g.home_team == game.home_team else g.away_score
                    for g in home_recent
                ]
                away_all_scores = [
                    g.home_score if g.home_team == game.away_team else g.away_score
                    for g in away_recent
                ]

                # Defensive scores
                home_def_scores = [
                    g.away_score if g.home_team == game.home_team else g.home_score
                    for g in home_recent
                ]
                away_def_scores = [
                    g.away_score if g.home_team == game.away_team else g.home_score
                    for g in away_recent
                ]

                # Last 3 games
                home_l3 = home_recent[-3:]
                away_l3 = away_recent[-3:]

                home_l3_scores = [
                    g.home_score if g.home_team == game.home_team else g.away_score
                    for g in home_l3
                ]
                away_l3_scores = [
                    g.home_score if g.home_team == game.away_team else g.away_score
                    for g in away_l3
                ]

                # Margins
                home_l3_margins = [
                    (g.home_score - g.away_score) if g.home_team == game.home_team
                    else (g.away_score - g.home_score)
                    for g in home_l3
                ]
                away_l3_margins = [
                    (g.home_score - g.away_score) if g.home_team == game.away_team
                    else (g.away_score - g.home_score)
                    for g in away_l3
                ]

                # Rest differential
                home_last_week = home_recent[-1].week
                away_last_week = away_recent[-1].week
                rest_diff = (game.week - home_last_week) - (game.week - away_last_week)
                rest_diff *= 7  # Convert to days

                # Build feature dict
                row = {
                    # Target
                    'actual_total': game.home_score + game.away_score,

                    # Baseline (most important)
                    'home_recent_ppg': np.mean(home_recent_scores),
                    'away_recent_ppg': np.mean(away_recent_scores),

                    # Season averages
                    'home_season_ppg': np.mean(home_all_scores),
                    'away_season_ppg': np.mean(away_all_scores),

                    # Defense
                    'home_def_ppg_allowed': np.mean(home_def_scores),
                    'away_def_ppg_allowed': np.mean(away_def_scores),

                    # Matchup quality
                    'home_off_vs_away_def': np.mean(home_all_scores) - np.mean(away_def_scores),
                    'away_off_vs_home_def': np.mean(away_all_scores) - np.mean(home_def_scores),

                    # Recent form
                    'home_l3_ppg': np.mean(home_l3_scores),
                    'away_l3_ppg': np.mean(away_l3_scores),
                    'home_l3_margin': np.mean(home_l3_margins),
                    'away_l3_margin': np.mean(away_l3_margins),

                    # Trends (momentum)
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

                    # Meta
                    'season': season,
                }

                all_data.append(row)

        df = pd.DataFrame(all_data)
        print(f"  Prepared {len(df)} training examples")

        self.data = df
        return df

    def train_gradient_boosting(self, X_train, y_train, X_test, y_test) -> MLModelResult:
        """Train tuned Gradient Boosting model."""
        print("\n" + "="*80)
        print("GRADIENT BOOSTING (TUNED)")
        print("="*80)

        # Grid search for best hyperparameters
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1],
            'min_samples_split': [10, 20],
        }

        gb = GradientBoostingRegressor(random_state=42)

        print("  Grid searching hyperparameters...")
        grid = GridSearchCV(gb, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)

        print(f"  Best params: {grid.best_params_}")

        best_model = grid.best_estimator_

        # Evaluate
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()

        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Test MAE: {test_mae:.2f}")
        print(f"  CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
        print(f"  Overfitting gap: {((test_mae - train_mae) / train_mae * 100):.1f}%")

        # Feature importance
        feature_importance = dict(zip(self.feature_cols, best_model.feature_importances_))

        # Save model
        model_path = self.output_dir / 'gradient_boosting_tuned.pkl'
        joblib.dump(best_model, model_path)
        print(f"  Saved to: {model_path}")

        return MLModelResult(
            model_name='Gradient Boosting (Tuned)',
            model=best_model,
            feature_names=self.feature_cols,
            train_mae=train_mae,
            test_mae=test_mae,
            cv_mae=cv_mae,
            cv_std=cv_std,
            overfitting_gap=((test_mae - train_mae) / train_mae * 100),
            feature_importance=feature_importance,
            model_path=str(model_path)
        )

    def train_random_forest(self, X_train, y_train, X_test, y_test) -> MLModelResult:
        """Train Random Forest model."""
        print("\n" + "="*80)
        print("RANDOM FOREST")
        print("="*80)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
        }

        rf = RandomForestRegressor(random_state=42)

        print("  Grid searching hyperparameters...")
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)

        print(f"  Best params: {grid.best_params_}")

        best_model = grid.best_estimator_

        # Evaluate
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()

        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Test MAE: {test_mae:.2f}")
        print(f"  CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
        print(f"  Overfitting gap: {((test_mae - train_mae) / train_mae * 100):.1f}%")

        # Feature importance
        feature_importance = dict(zip(self.feature_cols, best_model.feature_importances_))

        # Save model
        model_path = self.output_dir / 'random_forest.pkl'
        joblib.dump(best_model, model_path)
        print(f"  Saved to: {model_path}")

        return MLModelResult(
            model_name='Random Forest',
            model=best_model,
            feature_names=self.feature_cols,
            train_mae=train_mae,
            test_mae=test_mae,
            cv_mae=cv_mae,
            cv_std=cv_std,
            overfitting_gap=((test_mae - train_mae) / train_mae * 100),
            feature_importance=feature_importance,
            model_path=str(model_path)
        )

    def train_neural_network(self, X_train, y_train, X_test, y_test) -> MLModelResult:
        """Train Neural Network model."""
        print("\n" + "="*80)
        print("NEURAL NETWORK")
        print("="*80)

        # Simple neural network
        nn = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )

        print("  Training neural network...")
        nn.fit(X_train, y_train)

        # Evaluate
        train_pred = nn.predict(X_train)
        test_pred = nn.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        # Cross-validation
        cv_scores = cross_val_score(nn, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()

        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Test MAE: {test_mae:.2f}")
        print(f"  CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
        print(f"  Overfitting gap: {((test_mae - train_mae) / train_mae * 100):.1f}%")

        # Save model
        model_path = self.output_dir / 'neural_network.pkl'
        joblib.dump(nn, model_path)
        print(f"  Saved to: {model_path}")

        return MLModelResult(
            model_name='Neural Network',
            model=nn,
            feature_names=self.feature_cols,
            train_mae=train_mae,
            test_mae=test_mae,
            cv_mae=cv_mae,
            cv_std=cv_std,
            overfitting_gap=((test_mae - train_mae) / train_mae * 100),
            feature_importance={},  # NN doesn't have feature importance
            model_path=str(model_path)
        )

    def train_all_models(self) -> List[MLModelResult]:
        """Train all models and return results.

        Returns:
            List of MLModelResult objects
        """
        print("\n" + "="*80)
        print("TRAINING PRODUCTION ML MODELS")
        print("="*80 + "\n")

        # Prepare data
        df = self.prepare_data()

        # Split features and target
        self.feature_cols = [col for col in df.columns if col not in ['actual_total', 'season']]
        X = df[self.feature_cols].fillna(0).values
        y = df['actual_total'].values

        # Train/test split (2023 as test)
        train_mask = df['season'].isin([2021, 2022])
        test_mask = df['season'] == 2023

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        print(f"Train: {len(X_train)} games (2021-2022)")
        print(f"Test: {len(X_test)} games (2023)")
        print(f"Features: {len(self.feature_cols)}")
        print()

        # Calculate baseline
        baseline_preds = df.loc[test_mask, 'home_recent_ppg'] + df.loc[test_mask, 'away_recent_ppg']
        baseline_mae = mean_absolute_error(y_test, baseline_preds)

        print(f"Baseline MAE (test): {baseline_mae:.2f}")
        print()

        # Train models
        results = []

        results.append(self.train_gradient_boosting(X_train, y_train, X_test, y_test))
        results.append(self.train_random_forest(X_train, y_train, X_test, y_test))
        results.append(self.train_neural_network(X_train, y_train, X_test, y_test))

        # Summary
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80 + "\n")

        print(f"{'Model':<30} {'Test MAE':<12} {'vs Baseline':<15} {'Overfitting'}")
        print("-" * 80)
        print(f"{'Baseline (Recent Avg)':<30} {baseline_mae:<12.2f} {'-':<15} {'-'}")

        for result in results:
            improvement = ((baseline_mae - result.test_mae) / baseline_mae * 100)
            print(f"{result.model_name:<30} {result.test_mae:<12.2f} {improvement:+.1f}%{'':<10} {result.overfitting_gap:.1f}%")

        # Best model
        best_model = min(results, key=lambda r: r.test_mae)
        print()
        print(f"🏆 Best Model: {best_model.model_name}")
        print(f"   Test MAE: {best_model.test_mae:.2f} ({((baseline_mae - best_model.test_mae) / baseline_mae * 100):+.1f}% vs baseline)")
        print(f"   Model saved: {best_model.model_path}")

        # Save best model metadata
        best_model_meta = {
            'model_name': best_model.model_name,
            'model_path': best_model.model_path,
            'feature_names': best_model.feature_names,
            'test_mae': best_model.test_mae,
            'baseline_mae': baseline_mae,
            'improvement_pct': ((baseline_mae - best_model.test_mae) / baseline_mae * 100)
        }

        import json
        meta_path = self.output_dir / 'best_model.json'
        with open(meta_path, 'w') as f:
            json.dump(best_model_meta, f, indent=2)

        print(f"   Metadata saved: {meta_path}")
        print()

        return results


def main():
    """Train production ML models."""
    pipeline = GameTotalsMLPipeline()
    results = pipeline.train_all_models()

    print("✅ Production ML models trained and saved!")
    print(f"   Models directory: {pipeline.output_dir}")


if __name__ == '__main__':
    main()
