"""Historical Backtesting Framework.

Framework for validating prediction models against historical NFL data.
Calculates actual impacts from historical games to replace static assumptions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
import json


@dataclass
class BacktestResult:
    """Results from a backtest analysis."""
    feature_name: str
    seasons_tested: List[int]
    sample_size: int

    # Accuracy metrics
    rmse: float = 0.0  # Root Mean Square Error
    mae: float = 0.0   # Mean Absolute Error
    correlation: float = 0.0
    r_squared: float = 0.0

    # Statistical confidence
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    p_value: float = 1.0

    # Calculated factors
    calculated_factors: Dict = field(default_factory=dict)
    original_factors: Dict = field(default_factory=dict)

    # Recommendations
    should_update: bool = False
    improvement_pct: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class HistoricalGame:
    """Historical game data for backtesting."""
    game_id: str
    season: int
    week: int
    game_date: datetime

    home_team: str
    away_team: str
    home_score: int
    away_score: int

    # Weather
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[str] = None

    # Context
    is_primetime: bool = False
    is_division_game: bool = False
    is_playoff: bool = False

    # Player stats (will be populated)
    player_stats: Dict[str, Dict] = field(default_factory=dict)
    injuries: List[Dict] = field(default_factory=list)


class BacktestingFramework:
    """Core framework for historical backtesting."""

    def __init__(self, seasons: List[int] = None):
        """Initialize backtesting framework.

        Args:
            seasons: List of seasons to test (e.g., [2020, 2021, 2022, 2023, 2024])
                    Default uses complete seasons only (2020-2024). 2025 can be included
                    but has partial data (weeks 1-12).
        """
        self.seasons = seasons or [2020, 2021, 2022, 2023, 2024]
        self.data_dir = Path('inputs/historical')
        self.results_dir = Path('outputs/backtesting')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Cached historical data
        self.games_cache: Dict[int, List[HistoricalGame]] = {}
        self.player_stats_cache: Dict[str, pd.DataFrame] = {}

    def load_historical_games(self, season: int) -> List[HistoricalGame]:
        """Load historical games for a season.

        Args:
            season: Season year

        Returns:
            List of HistoricalGame objects
        """
        if season in self.games_cache:
            return self.games_cache[season]

        # Check if we have historical data files
        games_file = self.data_dir / f'games_{season}.csv'

        if not games_file.exists():
            print(f"Warning: No historical game data for {season}")
            print(f"  Expected: {games_file}")
            print(f"  Will need to fetch from nfl-data-py or ESPN API")
            return []

        # Load games from CSV
        df = pd.read_csv(games_file)
        games = []

        for _, row in df.iterrows():
            game = HistoricalGame(
                game_id=row['game_id'],
                season=season,
                week=row['week'],
                game_date=pd.to_datetime(row['game_date']),
                home_team=row['home_team'],
                away_team=row['away_team'],
                home_score=row['home_score'],
                away_score=row['away_score'],
                temperature=row.get('temperature'),
                wind_speed=row.get('wind_speed'),
                precipitation=row.get('precipitation', 'none'),
                is_primetime=row.get('is_primetime', False),
                is_division_game=row.get('is_division_game', False),
                is_playoff=row.get('is_playoff', False)
            )
            games.append(game)

        self.games_cache[season] = games
        return games

    def load_player_stats(self, season: int, stat_type: str = 'all') -> pd.DataFrame:
        """Load historical player stats.

        Args:
            season: Season year
            stat_type: 'passing', 'rushing', 'receiving', or 'all'

        Returns:
            DataFrame with player stats
        """
        cache_key = f"{season}_{stat_type}"
        if cache_key in self.player_stats_cache:
            return self.player_stats_cache[cache_key]

        stats_file = self.data_dir / f'player_stats_{season}_{stat_type}.csv'

        if not stats_file.exists():
            print(f"Warning: No player stats for {season} ({stat_type})")
            print(f"  Expected: {stats_file}")
            return pd.DataFrame()

        df = pd.read_csv(stats_file)
        self.player_stats_cache[cache_key] = df
        return df

    def calculate_actual_impact(
        self,
        baseline: float,
        actual: float,
        confidence_threshold: float = 0.5
    ) -> Tuple[float, float]:
        """Calculate actual impact from historical data.

        Args:
            baseline: Baseline expectation
            actual: Actual result
            confidence_threshold: Minimum confidence required

        Returns:
            (impact, confidence)
        """
        if baseline == 0:
            return 0.0, 0.0

        impact = actual - baseline

        # Calculate confidence based on sample size and variance
        # This would be more sophisticated with actual variance data
        confidence = min(1.0, abs(impact) / (baseline * 0.5))

        if confidence < confidence_threshold:
            return 0.0, confidence

        return impact, confidence

    def calculate_metrics(
        self,
        predicted: List[float],
        actual: List[float]
    ) -> Dict[str, float]:
        """Calculate accuracy metrics.

        Args:
            predicted: List of predicted values
            actual: List of actual values

        Returns:
            Dictionary of metrics
        """
        predicted = np.array(predicted)
        actual = np.array(actual)

        # Remove any NaN values
        mask = ~(np.isnan(predicted) | np.isnan(actual))
        predicted = predicted[mask]
        actual = actual[mask]

        if len(predicted) == 0:
            return {
                'rmse': 0.0,
                'mae': 0.0,
                'correlation': 0.0,
                'r_squared': 0.0,
                'sample_size': 0
            }

        # Calculate metrics
        errors = predicted - actual
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))

        # Correlation and R²
        if np.std(predicted) > 0 and np.std(actual) > 0:
            correlation = np.corrcoef(predicted, actual)[0, 1]
            r_squared = correlation ** 2
        else:
            correlation = 0.0
            r_squared = 0.0

        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(correlation),
            'r_squared': float(r_squared),
            'sample_size': len(predicted)
        }

    def run_backtest(
        self,
        feature_name: str,
        prediction_func: Callable,
        actual_func: Callable,
        seasons: List[int] = None
    ) -> BacktestResult:
        """Run a backtest for a specific feature.

        Args:
            feature_name: Name of feature being tested
            prediction_func: Function that generates predictions
            actual_func: Function that retrieves actual results
            seasons: Seasons to test (defaults to self.seasons)

        Returns:
            BacktestResult object
        """
        test_seasons = seasons or self.seasons

        all_predicted = []
        all_actual = []

        for season in test_seasons:
            games = self.load_historical_games(season)

            for game in games:
                try:
                    predicted = prediction_func(game)
                    actual = actual_func(game)

                    if predicted is not None and actual is not None:
                        all_predicted.append(predicted)
                        all_actual.append(actual)

                except Exception as e:
                    print(f"Error processing {game.game_id}: {e}")
                    continue

        # Calculate metrics
        metrics = self.calculate_metrics(all_predicted, all_actual)

        result = BacktestResult(
            feature_name=feature_name,
            seasons_tested=test_seasons,
            sample_size=metrics['sample_size'],
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            correlation=metrics['correlation'],
            r_squared=metrics['r_squared']
        )

        return result

    def compare_models(
        self,
        model_a_predictions: List[float],
        model_b_predictions: List[float],
        actuals: List[float],
        model_a_name: str = "Original",
        model_b_name: str = "Optimized"
    ) -> Dict:
        """Compare two models.

        Args:
            model_a_predictions: Predictions from model A
            model_b_predictions: Predictions from model B
            actuals: Actual values
            model_a_name: Name of model A
            model_b_name: Name of model B

        Returns:
            Comparison results
        """
        metrics_a = self.calculate_metrics(model_a_predictions, actuals)
        metrics_b = self.calculate_metrics(model_b_predictions, actuals)

        # Calculate improvement
        rmse_improvement = ((metrics_a['rmse'] - metrics_b['rmse']) / metrics_a['rmse']) * 100
        mae_improvement = ((metrics_a['mae'] - metrics_b['mae']) / metrics_a['mae']) * 100

        return {
            model_a_name: metrics_a,
            model_b_name: metrics_b,
            'improvements': {
                'rmse_improvement_pct': rmse_improvement,
                'mae_improvement_pct': mae_improvement,
                'correlation_delta': metrics_b['correlation'] - metrics_a['correlation']
            },
            'winner': model_b_name if metrics_b['rmse'] < metrics_a['rmse'] else model_a_name
        }

    def save_results(self, result: BacktestResult, filename: str = None):
        """Save backtest results to file.

        Args:
            result: BacktestResult to save
            filename: Optional filename (defaults to feature_name)
        """
        if filename is None:
            filename = f"{result.feature_name.lower().replace(' ', '_')}_backtest.json"

        output_file = self.results_dir / filename

        # Convert to dict
        result_dict = {
            'feature_name': result.feature_name,
            'seasons_tested': result.seasons_tested,
            'sample_size': result.sample_size,
            'metrics': {
                'rmse': result.rmse,
                'mae': result.mae,
                'correlation': result.correlation,
                'r_squared': result.r_squared
            },
            'calculated_factors': result.calculated_factors,
            'original_factors': result.original_factors,
            'should_update': result.should_update,
            'improvement_pct': result.improvement_pct,
            'notes': result.notes,
            'generated_at': datetime.now().isoformat()
        }

        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"Saved backtest results to {output_file}")

    def generate_summary_report(self, results: List[BacktestResult]) -> str:
        """Generate summary report from multiple backtest results.

        Args:
            results: List of BacktestResult objects

        Returns:
            Markdown-formatted report
        """
        report = ["# Backtesting Summary Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"Seasons Tested: {self.seasons}\n")
        report.append("\n## Overview\n")

        # Summary table
        report.append("| Feature | Sample Size | RMSE | MAE | R² | Update? | Improvement |")
        report.append("|---------|-------------|------|-----|----|---------|-----------:|")

        for result in results:
            update_icon = "✅" if result.should_update else "❌"
            report.append(
                f"| {result.feature_name} | {result.sample_size} | "
                f"{result.rmse:.2f} | {result.mae:.2f} | {result.r_squared:.3f} | "
                f"{update_icon} | {result.improvement_pct:+.1f}% |"
            )

        report.append("\n## Detailed Results\n")

        for result in results:
            report.append(f"\n### {result.feature_name}\n")
            report.append(f"- **Sample Size:** {result.sample_size} games")
            report.append(f"- **RMSE:** {result.rmse:.2f}")
            report.append(f"- **MAE:** {result.mae:.2f}")
            report.append(f"- **Correlation:** {result.correlation:.3f}")
            report.append(f"- **R²:** {result.r_squared:.3f}")

            if result.should_update:
                report.append(f"\n**Recommendation:** Update factors ({result.improvement_pct:+.1f}% improvement)")

            if result.notes:
                report.append("\n**Notes:**")
                for note in result.notes:
                    report.append(f"- {note}")

            if result.calculated_factors:
                report.append("\n**Calculated Factors:**")
                report.append("```json")
                report.append(json.dumps(result.calculated_factors, indent=2))
                report.append("```")

        return "\n".join(report)


# Singleton instance
backtesting_framework = BacktestingFramework()


if __name__ == "__main__":
    # Test framework
    framework = BacktestingFramework(seasons=[2022, 2023])

    print("Backtesting Framework initialized")
    print(f"Testing seasons: {framework.seasons}")
    print(f"Data directory: {framework.data_dir}")
    print(f"Results directory: {framework.results_dir}")

    # Try loading games
    for season in framework.seasons:
        games = framework.load_historical_games(season)
        print(f"\n{season} season: {len(games)} games loaded")
