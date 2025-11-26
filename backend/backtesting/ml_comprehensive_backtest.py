"""Machine Learning Model for Game Totals with Comprehensive Features.

Tests whether ML can find useful patterns in:
- Team offense/defense ratings
- Recent form
- Defense matchups (offense vs defense)
- Weather conditions
- Injuries
- Rest differential
- Home field advantage
- Division games, primetime, dome

Uses Gradient Boosting to learn optimal feature weights.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from backend.backtesting.framework import BacktestingFramework, BacktestResult


@dataclass
class GameFeatures:
    """Comprehensive features for a game."""
    game_id: str
    actual_total: float

    # Baseline
    home_recent_ppg: float
    away_recent_ppg: float

    # Offense vs Defense matchups
    home_off_ppg: float
    home_def_ppg: float
    away_off_ppg: float
    away_def_ppg: float

    # Matchup quality
    home_off_vs_away_def: float  # Home offense vs away defense
    away_off_vs_home_def: float  # Away offense vs home defense

    # Recent form (last 3 games)
    home_l3_ppg: float
    away_l3_ppg: float
    home_l3_margin: float
    away_l3_margin: float

    # Situational
    home_field_advantage: float = 2.5  # NFL average
    is_division_game: int = 0
    is_primetime: int = 0
    is_dome: int = 0

    # Rest
    rest_differential: int = 0  # Days (home - away)
    home_off_bye: int = 0
    away_off_bye: int = 0

    # Weather
    temperature: float = 70.0
    wind_speed: float = 0.0
    is_cold: int = 0  # < 32°F
    is_windy: int = 0  # > 15 MPH

    # Week
    week: int = 1


class MLGameTotalsBacktester:
    """ML-based game totals prediction with comprehensive features."""

    def __init__(self, framework: BacktestingFramework):
        """Initialize backtester.

        Args:
            framework: Backtesting framework
        """
        self.framework = framework
        self.features_list: List[GameFeatures] = []

    def run_backtest(self) -> BacktestResult:
        """Run ML backtest with cross-validation.

        Returns:
            BacktestResult
        """
        print("\n" + "="*80)
        print("MACHINE LEARNING GAME TOTALS BACKTEST")
        print("="*80 + "\n")

        print("Approach:")
        print("  1. Extract comprehensive features for each game")
        print("  2. Train Gradient Boosting model")
        print("  3. Cross-validate (5-fold) to prevent overfitting")
        print("  4. Compare to simple baseline")
        print()

        # Collect features
        print("Collecting features...")
        self._collect_all_features()

        if len(self.features_list) == 0:
            print("No features collected!")
            return BacktestResult(
                feature_name="ML Comprehensive",
                seasons_tested=self.framework.seasons,
                sample_size=0,
                mae=0.0
            )

        print(f"  ✓ Collected {len(self.features_list)} games")
        print()

        # Convert to DataFrame
        df = self._features_to_dataframe()

        # Show feature summary
        print("Feature Summary:")
        print(f"  Total features: {len(df.columns) - 2}")  # Exclude game_id, actual_total
        print()

        # Split features and target
        feature_cols = [col for col in df.columns if col not in ['game_id', 'actual_total']]
        X = df[feature_cols].fillna(0).values  # Fill NaN with 0
        y = df['actual_total'].values

        # Calculate baseline MAE (simple averaging)
        baseline_preds = df['home_recent_ppg'] + df['away_recent_ppg']
        baseline_mae = mean_absolute_error(y, baseline_preds)

        print(f"Baseline MAE (recent averaging): {baseline_mae:.2f}")
        print()

        # Cross-validation
        print("Cross-Validation (5-fold):")
        print("-" * 40)

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Baseline for this fold
            baseline_preds_fold = baseline_preds.iloc[test_idx]
            baseline_mae_fold = mean_absolute_error(y_test, baseline_preds_fold)

            improvement = ((baseline_mae_fold - mae) / baseline_mae_fold) * 100

            fold_results.append({
                'mae': mae,
                'rmse': rmse,
                'baseline_mae': baseline_mae_fold,
                'improvement': improvement,
                'model': model
            })

            print(f"  Fold {fold_idx}: MAE={mae:.2f}, Baseline={baseline_mae_fold:.2f}, Improvement={improvement:+.1f}%")

        # Average results
        avg_mae = np.mean([r['mae'] for r in fold_results])
        avg_rmse = np.mean([r['rmse'] for r in fold_results])
        avg_baseline_mae = np.mean([r['baseline_mae'] for r in fold_results])
        avg_improvement = np.mean([r['improvement'] for r in fold_results])

        print()
        print("="*80)
        print("AVERAGE CROSS-VALIDATION RESULTS")
        print("="*80 + "\n")

        print(f"Baseline MAE:     {avg_baseline_mae:.2f} points")
        print(f"ML Model MAE:     {avg_mae:.2f} points")
        print(f"Improvement:      {avg_improvement:+.1f}%")
        print(f"ML Model RMSE:    {avg_rmse:.2f} points")
        print()

        # Feature importance (from last fold)
        print("="*80)
        print("FEATURE IMPORTANCE (Top 15)")
        print("="*80 + "\n")

        last_model = fold_results[-1]['model']
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': last_model.feature_importances_
        }).sort_values('importance', ascending=False)

        for idx, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")

        print()

        # Analysis
        print("="*80)
        print("ANALYSIS")
        print("="*80 + "\n")

        if avg_improvement > 2.0:
            print("✅ ML model shows SIGNIFICANT improvement over baseline!")
            print(f"   Adding complex features helps by {avg_improvement:.1f}%")
        elif avg_improvement > 0.5:
            print("⚠️  ML model shows MARGINAL improvement over baseline")
            print(f"   Complex features add {avg_improvement:.1f}% value")
        elif avg_improvement > -0.5:
            print("❌ ML model shows NO improvement over baseline")
            print("   Complex features provide zero value")
        else:
            print("❌ ML model is WORSE than baseline")
            print(f"   Complex features hurt accuracy by {-avg_improvement:.1f}%")
            print("   Likely overfitting or noise")

        print()

        # Check for overfitting
        print("Overfitting Check:")
        train_scores = []
        test_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)

            train_scores.append(train_mae)
            test_scores.append(test_mae)

        avg_train_mae = np.mean(train_scores)
        avg_test_mae = np.mean(test_scores)
        gap = avg_test_mae - avg_train_mae

        print(f"  Train MAE: {avg_train_mae:.2f}")
        print(f"  Test MAE:  {avg_test_mae:.2f}")
        print(f"  Gap:       {gap:.2f} ({gap/avg_train_mae*100:.1f}%)")

        if gap / avg_train_mae > 0.20:
            print("  ⚠️  Model is overfitting (20%+ gap)")
        elif gap / avg_train_mae > 0.10:
            print("  ⚠️  Model shows some overfitting (10-20% gap)")
        else:
            print("  ✅ Model is not overfitting (<10% gap)")

        print()

        # Return result
        return BacktestResult(
            feature_name="ML Comprehensive",
            seasons_tested=self.framework.seasons,
            sample_size=len(self.features_list),
            mae=avg_mae,
            rmse=avg_rmse,
            improvement_pct=avg_improvement,
            notes=[
                f"Baseline: {avg_baseline_mae:.2f} MAE",
                f"ML Model: {avg_mae:.2f} MAE",
                f"Improvement: {avg_improvement:+.1f}%",
                f"Train-Test Gap: {gap:.2f} ({gap/avg_train_mae*100:.1f}%)"
            ]
        )

    def _collect_all_features(self):
        """Collect comprehensive features for all games."""
        for season in self.framework.seasons:
            games = self.framework.load_historical_games(season)

            for game in games:
                # Skip early season
                if game.week < 5:
                    continue

                # Skip if no scores
                if game.home_score is None or game.away_score is None:
                    continue

                # Extract features
                features = self._extract_game_features(game, games)

                if features is not None:
                    self.features_list.append(features)

    def _extract_game_features(self, game, all_games) -> Optional[GameFeatures]:
        """Extract all features for a game.

        Args:
            game: Game to extract features for
            all_games: All games in season

        Returns:
            GameFeatures or None if not enough data
        """
        # Get recent games for each team
        home_recent = [
            g for g in all_games
            if (g.home_team == game.home_team or g.away_team == game.home_team) and
            g.week < game.week and
            g.home_score is not None
        ]

        away_recent = [
            g for g in all_games
            if (g.home_team == game.away_team or g.away_team == game.away_team) and
            g.week < game.week and
            g.home_score is not None
        ]

        if len(home_recent) < 3 or len(away_recent) < 3:
            return None

        # Recent averaging (last 4 games)
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

        home_recent_ppg = np.mean(home_recent_scores)
        away_recent_ppg = np.mean(away_recent_scores)

        # Season averages (all games up to this point)
        home_all_scores = [
            g.home_score if g.home_team == game.home_team else g.away_score
            for g in home_recent
        ]
        away_all_scores = [
            g.home_score if g.home_team == game.away_team else g.away_score
            for g in away_recent
        ]

        home_off_ppg = np.mean(home_all_scores)
        away_off_ppg = np.mean(away_all_scores)

        # Defensive PPG (points allowed)
        home_def_ppg = np.mean([
            g.away_score if g.home_team == game.home_team else g.home_score
            for g in home_recent
        ])
        away_def_ppg = np.mean([
            g.away_score if g.home_team == game.away_team else g.home_score
            for g in away_recent
        ])

        # Matchup quality
        home_off_vs_away_def = home_off_ppg - away_def_ppg
        away_off_vs_home_def = away_off_ppg - home_def_ppg

        # Last 3 games form
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

        home_l3_ppg = np.mean(home_l3_scores)
        away_l3_ppg = np.mean(away_l3_scores)

        # Margins
        home_l3_margin = np.mean([
            (g.home_score - g.away_score) if g.home_team == game.home_team
            else (g.away_score - g.home_score)
            for g in home_l3
        ])
        away_l3_margin = np.mean([
            (g.home_score - g.away_score) if g.home_team == game.away_team
            else (g.away_score - g.home_score)
            for g in away_l3
        ])

        # Rest differential
        home_last_week = home_recent[-1].week
        away_last_week = away_recent[-1].week
        rest_diff = (game.week - home_last_week) - (game.week - away_last_week)
        rest_diff *= 7  # Convert to days

        # Bye weeks
        home_off_bye = 1 if (game.week - home_last_week) > 1 else 0
        away_off_bye = 1 if (game.week - away_last_week) > 1 else 0

        # Weather
        temperature = game.temperature if game.temperature is not None else 70.0
        wind_speed = game.wind_speed if game.wind_speed is not None else 0.0
        is_cold = 1 if temperature < 32 else 0
        is_windy = 1 if wind_speed > 15 else 0

        # Situational
        is_primetime = 1 if game.is_primetime else 0
        is_division_game = 1 if game.is_division_game else 0
        is_dome = 1 if hasattr(game, 'is_dome') and game.is_dome else 0

        return GameFeatures(
            game_id=game.game_id,
            actual_total=game.home_score + game.away_score,
            home_recent_ppg=home_recent_ppg,
            away_recent_ppg=away_recent_ppg,
            home_off_ppg=home_off_ppg,
            home_def_ppg=home_def_ppg,
            away_off_ppg=away_off_ppg,
            away_def_ppg=away_def_ppg,
            home_off_vs_away_def=home_off_vs_away_def,
            away_off_vs_home_def=away_off_vs_home_def,
            home_l3_ppg=home_l3_ppg,
            away_l3_ppg=away_l3_ppg,
            home_l3_margin=home_l3_margin,
            away_l3_margin=away_l3_margin,
            rest_differential=rest_diff,
            home_off_bye=home_off_bye,
            away_off_bye=away_off_bye,
            temperature=temperature,
            wind_speed=wind_speed,
            is_cold=is_cold,
            is_windy=is_windy,
            is_primetime=is_primetime,
            is_division_game=is_division_game,
            is_dome=is_dome,
            week=game.week
        )

    def _features_to_dataframe(self) -> pd.DataFrame:
        """Convert features to DataFrame.

        Returns:
            DataFrame with all features
        """
        data = []

        for features in self.features_list:
            row = {
                'game_id': features.game_id,
                'actual_total': features.actual_total,
                'home_recent_ppg': features.home_recent_ppg,
                'away_recent_ppg': features.away_recent_ppg,
                'home_off_ppg': features.home_off_ppg,
                'home_def_ppg': features.home_def_ppg,
                'away_off_ppg': features.away_off_ppg,
                'away_def_ppg': features.away_def_ppg,
                'home_off_vs_away_def': features.home_off_vs_away_def,
                'away_off_vs_home_def': features.away_off_vs_home_def,
                'home_l3_ppg': features.home_l3_ppg,
                'away_l3_ppg': features.away_l3_ppg,
                'home_l3_margin': features.home_l3_margin,
                'away_l3_margin': features.away_l3_margin,
                'home_field_advantage': features.home_field_advantage,
                'rest_differential': features.rest_differential,
                'home_off_bye': features.home_off_bye,
                'away_off_bye': features.away_off_bye,
                'temperature': features.temperature,
                'wind_speed': features.wind_speed,
                'is_cold': features.is_cold,
                'is_windy': features.is_windy,
                'is_primetime': features.is_primetime,
                'is_division_game': features.is_division_game,
                'is_dome': features.is_dome,
                'week': features.week,
            }
            data.append(row)

        return pd.DataFrame(data)


def main():
    """Run ML comprehensive backtest."""
    framework = BacktestingFramework(seasons=[2021, 2022, 2023])
    backtester = MLGameTotalsBacktester(framework)
    result = backtester.run_backtest()

    print(f"\n✅ ML Backtest complete")
    print(f"   MAE: {result.mae:.2f} ({result.improvement_pct:+.1f}%)")


if __name__ == '__main__':
    main()
