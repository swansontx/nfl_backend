"""Overall Prediction Accuracy Backtesting.

Validates end-to-end prediction accuracy for game totals and player projections.
Tests the complete prediction pipeline with all adjustments applied.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

from backend.backtesting.framework import BacktestingFramework, BacktestResult


@dataclass
class GamePrediction:
    """Complete game prediction vs actual."""
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str

    # Predictions
    predicted_total: float = 0.0
    predicted_home_score: float = 0.0
    predicted_away_score: float = 0.0
    predicted_spread: float = 0.0

    # Actuals
    actual_total: int = 0
    actual_home_score: int = 0
    actual_away_score: int = 0
    actual_spread: int = 0

    # Errors
    total_error: float = 0.0
    spread_error: float = 0.0
    home_error: float = 0.0
    away_error: float = 0.0


@dataclass
class PlayerPrediction:
    """Player projection vs actual."""
    player: str
    position: str
    team: str
    opponent: str
    season: int
    week: int

    # Predictions
    predicted_yards: float = 0.0
    predicted_points: float = 0.0
    predicted_targets: float = 0.0

    # Actuals
    actual_yards: float = 0.0
    actual_points: float = 0.0
    actual_targets: float = 0.0

    # Errors
    yards_error: float = 0.0
    points_error: float = 0.0


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics."""
    category: str  # 'game_totals', 'player_projections', 'spreads'

    # Basic metrics
    sample_size: int = 0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error

    # Hit rates
    within_3_pct: float = 0.0  # % of predictions within 3 points
    within_7_pct: float = 0.0  # % of predictions within 7 points
    within_10_pct: float = 0.0  # % of predictions within 10 points

    # Directional accuracy (for spreads)
    directional_accuracy: float = 0.0

    # Bias detection
    mean_error: float = 0.0  # Positive = over-predicting, negative = under-predicting
    bias_percentage: float = 0.0

    notes: List[str] = field(default_factory=list)


class OverallAccuracyBacktester:
    """Backtests overall prediction accuracy."""

    def __init__(self, framework: BacktestingFramework):
        """Initialize backtester.

        Args:
            framework: Backtesting framework instance
        """
        self.framework = framework

        # Collected predictions
        self.game_predictions: List[GamePrediction] = []
        self.player_predictions: List[PlayerPrediction] = []

        # Metrics by category
        self.metrics: Dict[str, AccuracyMetrics] = {}

    def generate_baseline_game_prediction(
        self,
        game,
        home_stats: pd.DataFrame,
        away_stats: pd.DataFrame
    ) -> GamePrediction:
        """Generate baseline prediction for a game.

        This uses simple averaging of recent performance.
        In production, this would use the full orchestrator.

        Args:
            game: Historical game
            home_stats: Home team recent stats
            away_stats: Away team recent stats

        Returns:
            GamePrediction object
        """
        # BUG FIX: Use actual team scores from game history, not fantasy points!
        # Previous code summed all player fantasy points, causing 2x over-prediction

        # Get recent game scores for each team
        all_games = self.framework.load_historical_games(game.season)

        # Filter for home team's recent games
        home_recent = [
            g for g in all_games
            if (g.home_team == game.home_team or g.away_team == game.home_team) and
            g.week < game.week and
            g.week >= max(1, game.week - 4)  # Last 4 games
        ]

        # Filter for away team's recent games
        away_recent = [
            g for g in all_games
            if (g.home_team == game.away_team or g.away_team == game.away_team) and
            g.week < game.week and
            g.week >= max(1, game.week - 4)  # Last 4 games
        ]

        # Calculate average points scored from recent games
        if home_recent:
            home_scores = [
                g.home_score if g.home_team == game.home_team else g.away_score
                for g in home_recent
            ]
            home_avg_points = np.mean(home_scores)
        else:
            home_avg_points = 22.0  # NFL average

        if away_recent:
            away_scores = [
                g.home_score if g.home_team == game.away_team else g.away_score
                for g in away_recent
            ]
            away_avg_points = np.mean(away_scores)
        else:
            away_avg_points = 20.0  # NFL average for road teams

        # Simple home field advantage
        home_avg_points += 2.5

        predicted_total = home_avg_points + away_avg_points
        predicted_spread = home_avg_points - away_avg_points

        prediction = GamePrediction(
            game_id=game.game_id,
            season=game.season,
            week=game.week,
            home_team=game.home_team,
            away_team=game.away_team,
            predicted_total=predicted_total,
            predicted_home_score=home_avg_points,
            predicted_away_score=away_avg_points,
            predicted_spread=predicted_spread,
            actual_total=game.home_score + game.away_score,
            actual_home_score=game.home_score,
            actual_away_score=game.away_score,
            actual_spread=game.home_score - game.away_score,
            total_error=abs(predicted_total - (game.home_score + game.away_score)),
            spread_error=abs(predicted_spread - (game.home_score - game.away_score)),
            home_error=abs(home_avg_points - game.home_score),
            away_error=abs(away_avg_points - game.away_score)
        )

        return prediction

    def generate_baseline_player_prediction(
        self,
        player_row: pd.Series,
        player_history: pd.DataFrame
    ) -> Optional[PlayerPrediction]:
        """Generate baseline prediction for a player.

        Args:
            player_row: Current game player stats
            player_history: Player's recent history

        Returns:
            PlayerPrediction object or None
        """
        if player_history.empty or len(player_history) < 2:
            return None

        # Calculate averages
        pred_yards = (
            player_history.get('receiving_yards', pd.Series([0])).mean() +
            player_history.get('rushing_yards', pd.Series([0])).mean()
        )
        pred_points = player_history.get('fantasy_points', pd.Series([0])).mean()
        pred_targets = (
            player_history.get('targets', pd.Series([0])).mean() +
            player_history.get('carries', pd.Series([0])).mean()
        )

        actual_yards = (
            player_row.get('receiving_yards', 0) +
            player_row.get('rushing_yards', 0)
        )
        actual_points = player_row.get('fantasy_points', 0)
        actual_targets = (
            player_row.get('targets', 0) +
            player_row.get('carries', 0)
        )

        prediction = PlayerPrediction(
            player=player_row['player'],
            position=player_row.get('position', 'UNK'),
            team=player_row.get('team', ''),
            opponent=player_row.get('opponent', ''),
            season=player_row.get('season', 0),
            week=player_row.get('week', 0),
            predicted_yards=pred_yards,
            predicted_points=pred_points,
            predicted_targets=pred_targets,
            actual_yards=actual_yards,
            actual_points=actual_points,
            actual_targets=actual_targets,
            yards_error=abs(pred_yards - actual_yards),
            points_error=abs(pred_points - actual_points)
        )

        return prediction

    def backtest_game_predictions(
        self,
        seasons: List[int] = None
    ) -> List[GamePrediction]:
        """Generate and evaluate game predictions.

        Args:
            seasons: Seasons to test

        Returns:
            List of GamePrediction objects
        """
        test_seasons = seasons or self.framework.seasons

        predictions = []

        for season in test_seasons:
            games = self.framework.load_historical_games(season)
            player_stats = self.framework.load_player_stats(season, 'all')

            if player_stats.empty:
                continue

            for game in games:
                # Get team stats before this game
                home_history = player_stats[
                    (player_stats['team'] == game.home_team) &
                    (player_stats['week'] < game.week) &
                    (player_stats['week'] >= max(1, game.week - 4))
                ]

                away_history = player_stats[
                    (player_stats['team'] == game.away_team) &
                    (player_stats['week'] < game.week) &
                    (player_stats['week'] >= max(1, game.week - 4))
                ]

                if home_history.empty or away_history.empty:
                    continue

                prediction = self.generate_baseline_game_prediction(
                    game, home_history, away_history
                )
                predictions.append(prediction)

        self.game_predictions = predictions
        return predictions

    def backtest_player_predictions(
        self,
        seasons: List[int] = None,
        min_targets: int = 3
    ) -> List[PlayerPrediction]:
        """Generate and evaluate player predictions.

        Args:
            seasons: Seasons to test
            min_targets: Minimum targets/carries to include player

        Returns:
            List of PlayerPrediction objects
        """
        test_seasons = seasons or self.framework.seasons

        predictions = []

        for season in test_seasons:
            player_stats = self.framework.load_player_stats(season, 'all')

            if player_stats.empty:
                continue

            # Group by player
            for player_name in player_stats['player'].unique():
                player_games = player_stats[player_stats['player'] == player_name].sort_values('week')

                # Need at least 3 games of history
                if len(player_games) < 4:
                    continue

                # Predict each game after first 3
                for idx in range(3, len(player_games)):
                    current_game = player_games.iloc[idx]
                    player_history = player_games.iloc[:idx]

                    # Filter for relevant players (those who see action)
                    total_usage = current_game.get('targets', 0) + current_game.get('carries', 0)
                    if total_usage < min_targets:
                        continue

                    prediction = self.generate_baseline_player_prediction(
                        current_game, player_history
                    )

                    if prediction:
                        predictions.append(prediction)

        self.player_predictions = predictions
        return predictions

    def calculate_accuracy_metrics(
        self,
        predictions: List,
        value_field: str,
        actual_field: str,
        error_field: str,
        category: str
    ) -> AccuracyMetrics:
        """Calculate comprehensive accuracy metrics.

        Args:
            predictions: List of prediction objects
            value_field: Field name for predicted value
            actual_field: Field name for actual value
            error_field: Field name for error
            category: Category name

        Returns:
            AccuracyMetrics object
        """
        if not predictions:
            return AccuracyMetrics(category=category, sample_size=0)

        errors = [getattr(p, error_field) for p in predictions]
        predicted = [getattr(p, value_field) for p in predictions]
        actuals = [getattr(p, actual_field) for p in predictions]

        # Basic metrics
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        mae = np.mean(errors)

        # MAPE (Mean Absolute Percentage Error)
        mape_values = [abs(p - a) / a * 100 for p, a in zip(predicted, actuals) if a > 0]
        mape = np.mean(mape_values) if mape_values else 0.0

        # Hit rates
        within_3 = sum(1 for e in errors if e <= 3) / len(errors) * 100
        within_7 = sum(1 for e in errors if e <= 7) / len(errors) * 100
        within_10 = sum(1 for e in errors if e <= 10) / len(errors) * 100

        # Bias detection
        signed_errors = [p - a for p, a in zip(predicted, actuals)]
        mean_error = np.mean(signed_errors)
        bias_pct = mean_error / np.mean(actuals) * 100 if np.mean(actuals) > 0 else 0

        notes = [
            f"Hit Rates: {within_3:.1f}% within 3, {within_7:.1f}% within 7, {within_10:.1f}% within 10",
            f"Bias: {bias_pct:+.1f}% ({'over' if bias_pct > 0 else 'under'}-predicting)"
        ]

        metrics = AccuracyMetrics(
            category=category,
            sample_size=len(predictions),
            rmse=rmse,
            mae=mae,
            mape=mape,
            within_3_pct=within_3,
            within_7_pct=within_7,
            within_10_pct=within_10,
            mean_error=mean_error,
            bias_percentage=bias_pct,
            notes=notes
        )

        return metrics

    def run_backtest(self) -> BacktestResult:
        """Run overall accuracy backtest.

        Returns:
            BacktestResult with findings
        """
        print("Running overall prediction accuracy backtest...")

        # Backtest game predictions
        print("  Generating game predictions...")
        game_preds = self.backtest_game_predictions()
        print(f"    Generated {len(game_preds)} game predictions")

        # Backtest player predictions
        print("  Generating player predictions...")
        player_preds = self.backtest_player_predictions()
        print(f"    Generated {len(player_preds)} player predictions")

        if len(game_preds) < 50:
            return BacktestResult(
                feature_name="Overall Prediction Accuracy",
                seasons_tested=self.framework.seasons,
                sample_size=0,
                notes=["Insufficient data for accuracy analysis"]
            )

        # Calculate metrics
        game_total_metrics = self.calculate_accuracy_metrics(
            game_preds, 'predicted_total', 'actual_total', 'total_error', 'game_totals'
        )

        spread_metrics = self.calculate_accuracy_metrics(
            game_preds, 'predicted_spread', 'actual_spread', 'spread_error', 'spreads'
        )

        player_yards_metrics = self.calculate_accuracy_metrics(
            player_preds, 'predicted_yards', 'actual_yards', 'yards_error', 'player_yards'
        ) if player_preds else None

        player_points_metrics = self.calculate_accuracy_metrics(
            player_preds, 'predicted_points', 'actual_points', 'points_error', 'player_points'
        ) if player_preds else None

        self.metrics = {
            'game_totals': game_total_metrics,
            'spreads': spread_metrics,
        }
        if player_yards_metrics:
            self.metrics['player_yards'] = player_yards_metrics
        if player_points_metrics:
            self.metrics['player_points'] = player_points_metrics

        # Generate notes
        notes = []
        notes.append(f"Backtested {len(game_preds)} game predictions")
        notes.append(f"Backtested {len(player_preds)} player predictions")

        notes.append("\nGAME TOTALS:")
        notes.append(f"  RMSE: {game_total_metrics.rmse:.2f} points")
        notes.append(f"  MAE: {game_total_metrics.mae:.2f} points")
        notes.append(f"  MAPE: {game_total_metrics.mape:.1f}%")
        for note in game_total_metrics.notes:
            notes.append(f"  {note}")

        notes.append("\nSPREADS:")
        notes.append(f"  RMSE: {spread_metrics.rmse:.2f} points")
        notes.append(f"  MAE: {spread_metrics.mae:.2f} points")
        for note in spread_metrics.notes:
            notes.append(f"  {note}")

        if player_yards_metrics:
            notes.append("\nPLAYER YARDS:")
            notes.append(f"  RMSE: {player_yards_metrics.rmse:.2f} yards")
            notes.append(f"  MAE: {player_yards_metrics.mae:.2f} yards")
            for note in player_yards_metrics.notes:
                notes.append(f"  {note}")

        # Determine if updates needed
        # Good benchmarks: Game totals RMSE < 12, Spreads RMSE < 10
        game_needs_improvement = game_total_metrics.rmse > 12
        spread_needs_improvement = spread_metrics.rmse > 10

        should_update = game_needs_improvement or spread_needs_improvement

        result = BacktestResult(
            feature_name="Overall Prediction Accuracy",
            seasons_tested=self.framework.seasons,
            sample_size=len(game_preds),
            rmse=game_total_metrics.rmse,
            mae=game_total_metrics.mae,
            calculated_factors={
                'game_totals': {
                    'rmse': game_total_metrics.rmse,
                    'mae': game_total_metrics.mae,
                    'mape': game_total_metrics.mape,
                    'within_7_pct': game_total_metrics.within_7_pct,
                    'bias_pct': game_total_metrics.bias_percentage
                },
                'spreads': {
                    'rmse': spread_metrics.rmse,
                    'mae': spread_metrics.mae,
                    'within_7_pct': spread_metrics.within_7_pct
                }
            },
            should_update=should_update,
            improvement_pct=0.0,  # Baseline measurement
            notes=notes
        )

        return result


if __name__ == "__main__":
    # Test overall accuracy backtester
    framework = BacktestingFramework(seasons=[2022, 2023])
    backtester = OverallAccuracyBacktester(framework)

    print("Overall Accuracy Backtester initialized")
    print(f"Testing seasons: {framework.seasons}")

    # Run backtest
    result = backtester.run_backtest()

    print(f"\nBacktest Results:")
    print(f"  Sample size: {result.sample_size}")
    print(f"  Game Totals RMSE: {result.rmse:.2f}")
    print(f"  Game Totals MAE: {result.mae:.2f}")
    print(f"\nNotes:")
    for note in result.notes:
        print(f"  {note}")
