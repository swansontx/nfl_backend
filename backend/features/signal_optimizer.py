"""
Signal Weight Optimizer

Learns optimal weights for all prediction signals from historical data.
Uses cross-validation and regression to find weights that maximize
prediction accuracy.

NO HARDCODED ESTIMATES - everything learned from data!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from backend.features.contextual_signals import ContextualSignalsExtractor
from backend.analysis.advanced_team_metrics import AdvancedTeamMetricsCalculator
from pathlib import Path


@dataclass
class OptimizedWeights:
    """Container for learned signal weights."""

    # Team metric weights (for spread)
    turnover_margin_weight: float = 0.0
    epa_differential_weight: float = 0.0
    success_rate_weight: float = 0.0
    red_zone_weight: float = 0.0

    # Team metric weights (for total)
    pace_weight: float = 0.0
    explosive_play_weight: float = 0.0

    # Contextual signal weights
    wind_weight: float = 0.0
    rest_diff_weight: float = 0.0
    primetime_weight: float = 0.0
    divisional_weight: float = 0.0

    # Performance metrics
    spread_mae: float = 0.0
    total_mae: float = 0.0
    spread_ats_pct: float = 0.0
    total_ou_pct: float = 0.0


class SignalWeightOptimizer:
    """
    Learns optimal weights for all prediction signals.

    Uses historical game data to:
    1. Extract all available signals
    2. Build feature matrix
    3. Fit regression models to learn weights
    4. Cross-validate to prevent overfitting
    5. Return optimized weights
    """

    def __init__(self, season: int = 2025, inputs_dir: str = "inputs"):
        self.season = season
        self.inputs_dir = Path(inputs_dir)

        # Load data
        self.games_df = self._load_games()

        # Initialize signal extractors
        self.context_extractor = ContextualSignalsExtractor(season=season, inputs_dir=str(self.inputs_dir))

        pbp_file = self.inputs_dir / f"play_by_play_{season}.parquet"
        if pbp_file.exists():
            self.metrics_calculator = AdvancedTeamMetricsCalculator(season=season, pbp_file=pbp_file)
        else:
            print(f"Warning: No PBP data, team metrics unavailable")
            self.metrics_calculator = None

    def _load_games(self) -> pd.DataFrame:
        """Load completed games with results."""
        schedule_file = self.inputs_dir / "schedules_2024_2025.csv"

        df = pd.read_csv(schedule_file)
        df = df[
            (df['season'] == self.season) &
            (df['game_type'] == 'REG') &
            (df['away_score'].notna()) &
            (df['home_score'].notna())
        ].copy()

        # Calculate outcomes
        df['actual_spread'] = df['home_score'] - df['away_score']
        df['actual_total'] = df['home_score'] + df['away_score']

        return df

    def extract_all_signals(self, game_id: str, home_team: str, away_team: str, week: int) -> Dict:
        """
        Extract ALL available signals for a game.

        Returns dict with all signal values.
        """
        signals = {}

        # Contextual signals
        context = self.context_extractor.extract_game_context(game_id)

        signals['wind_speed'] = context.wind_speed if context.wind_speed else 0.0
        signals['temperature'] = context.temperature if context.temperature else 65.0
        signals['rest_differential'] = context.rest_differential
        signals['is_primetime'] = 1.0 if context.is_primetime else 0.0
        signals['is_divisional'] = 1.0 if context.is_divisional else 0.0
        signals['is_outdoor'] = 1.0 if context.roof_type == 'outdoors' else 0.0

        # Team metrics (if available)
        if self.metrics_calculator:
            try:
                # Get full season metrics
                home_metrics = self.metrics_calculator.calculate_team_metrics(home_team, None)
                away_metrics = self.metrics_calculator.calculate_team_metrics(away_team, None)

                # Differentials for spreads
                signals['turnover_margin_diff'] = home_metrics.get('turnover_margin', 0) - away_metrics.get('turnover_margin', 0)
                signals['epa_diff'] = (
                    home_metrics.get('epa_per_play_offense', 0) - away_metrics.get('epa_per_play_defense', 0)
                ) - (
                    away_metrics.get('epa_per_play_offense', 0) - home_metrics.get('epa_per_play_defense', 0)
                )
                signals['success_rate_diff'] = (
                    home_metrics.get('success_rate_offense', 0) - away_metrics.get('success_rate_defense', 0)
                ) - (
                    away_metrics.get('success_rate_offense', 0) - home_metrics.get('success_rate_defense', 0)
                )
                signals['red_zone_diff'] = home_metrics.get('red_zone_td_pct', 0) - away_metrics.get('red_zone_td_pct', 0)

                # Combined metrics for totals
                signals['combined_pace'] = (
                    home_metrics.get('plays_per_game', 65) + away_metrics.get('plays_per_game', 65)
                ) / 2.0
                signals['combined_explosive'] = (
                    home_metrics.get('explosive_play_rate', 0.1) + away_metrics.get('explosive_play_rate', 0.1)
                ) / 2.0

            except Exception as e:
                # If metrics fail, use defaults
                signals['turnover_margin_diff'] = 0.0
                signals['epa_diff'] = 0.0
                signals['success_rate_diff'] = 0.0
                signals['red_zone_diff'] = 0.0
                signals['combined_pace'] = 65.0
                signals['combined_explosive'] = 0.1
        else:
            # No metrics available
            signals['turnover_margin_diff'] = 0.0
            signals['epa_diff'] = 0.0
            signals['success_rate_diff'] = 0.0
            signals['red_zone_diff'] = 0.0
            signals['combined_pace'] = 65.0
            signals['combined_explosive'] = 0.1

        return signals

    def build_feature_matrix(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Build feature matrix from all games.

        Returns:
            (features_df, spread_targets, total_targets)
        """
        features_list = []
        spread_targets = []
        total_targets = []

        print(f"\nExtracting signals from {len(self.games_df)} games...")

        for idx, game in self.games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            week = game['week']

            # Extract signals
            signals = self.extract_all_signals(game_id, home_team, away_team, week)

            # Add baseline prediction (simple team strength)
            # For now, use league average as baseline
            signals['baseline_spread'] = 0.0  # Home field advantage built into other signals
            signals['baseline_total'] = 45.0  # League average

            features_list.append(signals)

            # Targets (what we're trying to predict)
            spread_targets.append(game['actual_spread'])
            total_targets.append(game['actual_total'])

        features_df = pd.DataFrame(features_list)
        spread_targets = pd.Series(spread_targets)
        total_targets = pd.Series(total_targets)

        return features_df, spread_targets, total_targets

    def _ridge_regression(self, X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        """
        Ridge regression using numpy.
        w = (X^T X + alpha * I)^-1 X^T y
        """
        n_features = X.shape[1]
        I = np.eye(n_features)

        # Ridge formula
        w = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y

        return w

    def optimize_spread_weights(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Learn optimal weights for spread prediction.

        Uses Ridge regression with cross-validation.
        """
        print("\n" + "="*70)
        print("OPTIMIZING SPREAD WEIGHTS")
        print("="*70)

        # Select features relevant to spreads
        spread_features = [
            'baseline_spread',
            'turnover_margin_diff',
            'epa_diff',
            'success_rate_diff',
            'red_zone_diff',
            'rest_differential',
            'is_primetime',
            'is_divisional'
        ]

        X = features[spread_features].fillna(0).values
        y = targets.values

        # Try different regularization strengths
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha = None
        best_score = float('inf')

        for alpha in alphas:
            # Manual cross-validation
            fold_size = len(X) // cv_folds
            scores = []

            for fold in range(cv_folds):
                # Create train/val split
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else len(X)

                val_idx = list(range(val_start, val_end))
                train_idx = list(range(0, val_start)) + list(range(val_end, len(X)))

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Fit model
                weights = self._ridge_regression(X_train, y_train, alpha)

                # Predict
                preds = X_val @ weights

                # Calculate MAE
                mae = np.mean(np.abs(y_val - preds))
                scores.append(mae)

            avg_score = np.mean(scores)

            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha

        # Train final model with best alpha
        final_weights = self._ridge_regression(X, y, best_alpha)

        # Extract weights
        weights = dict(zip(spread_features, final_weights))

        print(f"\nBest alpha: {best_alpha}")
        print(f"Cross-validated MAE: {best_score:.2f} points")
        print(f"\nLearned Weights:")
        for feature, weight in weights.items():
            if abs(weight) > 0.01:
                print(f"  {feature:30s}: {weight:+.4f}")

        return weights

    def optimize_total_weights(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Learn optimal weights for total prediction.

        Uses Ridge regression with cross-validation.
        """
        print("\n" + "="*70)
        print("OPTIMIZING TOTAL WEIGHTS")
        print("="*70)

        # Select features relevant to totals
        total_features = [
            'baseline_total',
            'combined_pace',
            'combined_explosive',
            'wind_speed',
            'temperature',
            'is_primetime',
            'is_divisional',
            'is_outdoor'
        ]

        X = features[total_features].fillna(0).values
        y = targets.values

        # Try different regularization strengths
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha = None
        best_score = float('inf')

        for alpha in alphas:
            # Manual cross-validation
            fold_size = len(X) // cv_folds
            scores = []

            for fold in range(cv_folds):
                # Create train/val split
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else len(X)

                val_idx = list(range(val_start, val_end))
                train_idx = list(range(0, val_start)) + list(range(val_end, len(X)))

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Fit model
                weights = self._ridge_regression(X_train, y_train, alpha)

                # Predict
                preds = X_val @ weights

                # Calculate MAE
                mae = np.mean(np.abs(y_val - preds))
                scores.append(mae)

            avg_score = np.mean(scores)

            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha

        # Train final model with best alpha
        final_weights = self._ridge_regression(X, y, best_alpha)

        # Extract weights
        weights = dict(zip(total_features, final_weights))

        print(f"\nBest alpha: {best_alpha}")
        print(f"Cross-validated MAE: {best_score:.2f} points")
        print(f"\nLearned Weights:")
        for feature, weight in weights.items():
            if abs(weight) > 0.01:
                print(f"  {feature:30s}: {weight:+.4f}")

        return weights

    def optimize_all_weights(self, cv_folds: int = 5) -> OptimizedWeights:
        """
        Learn optimal weights for all signals.

        Returns OptimizedWeights with learned parameters.
        """
        # Build feature matrix
        features, spread_targets, total_targets = self.build_feature_matrix()

        print(f"\nFeatures extracted: {len(features.columns)}")
        print(f"Games available: {len(features)}")

        # Optimize spread weights
        spread_weights = self.optimize_spread_weights(features, spread_targets, cv_folds)

        # Optimize total weights
        total_weights = self.optimize_total_weights(features, total_targets, cv_folds)

        # Package into OptimizedWeights
        return OptimizedWeights(
            turnover_margin_weight=spread_weights.get('turnover_margin_diff', 0.0),
            epa_differential_weight=spread_weights.get('epa_diff', 0.0),
            success_rate_weight=spread_weights.get('success_rate_diff', 0.0),
            red_zone_weight=spread_weights.get('red_zone_diff', 0.0),
            pace_weight=total_weights.get('combined_pace', 0.0),
            explosive_play_weight=total_weights.get('combined_explosive', 0.0),
            wind_weight=total_weights.get('wind_speed', 0.0),
            rest_diff_weight=spread_weights.get('rest_differential', 0.0),
            primetime_weight=total_weights.get('is_primetime', 0.0),
            divisional_weight=total_weights.get('is_divisional', 0.0)
        )


def learn_optimal_weights(season: int = 2025, inputs_dir: str = "inputs") -> OptimizedWeights:
    """
    Convenience function to learn optimal weights from data.

    Usage:
        from backend.features.signal_optimizer import learn_optimal_weights
        weights = learn_optimal_weights(season=2025)
        print(f"Turnover margin weight: {weights.turnover_margin_weight}")
    """
    optimizer = SignalWeightOptimizer(season=season, inputs_dir=inputs_dir)
    return optimizer.optimize_all_weights()
