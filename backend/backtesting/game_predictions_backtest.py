"""
Backtesting framework for game prediction improvements.

Compares baseline predictions (simple PPG) vs enhanced predictions
(with pace, turnovers, efficiency metrics) against actual results.

Usage:
    from backend.backtesting.game_predictions_backtest import GamePredictionsBacktest

    backtest = GamePredictionsBacktest(season=2024)
    results = backtest.run_backtest(weeks=range(1, 13))
    backtest.print_report(results)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from backend.analysis.game_markets import GameMarketAnalyzer


@dataclass
class PredictionResult:
    """Single game prediction result."""
    game_id: str
    week: int
    home_team: str
    away_team: str

    # Actual results
    actual_home_score: float
    actual_away_score: float
    actual_spread: float  # Positive = home won by this margin
    actual_total: float

    # Betting lines
    line_spread: Optional[float]  # Positive = home favored
    line_total: Optional[float]

    # Baseline predictions (without enhanced metrics)
    baseline_home_score: float
    baseline_away_score: float
    baseline_spread: float
    baseline_total: float

    # Enhanced predictions (with pace/turnover/efficiency)
    enhanced_home_score: float
    enhanced_away_score: float
    enhanced_spread: float
    enhanced_total: float

    # Accuracy metrics
    baseline_spread_error: float
    enhanced_spread_error: float
    baseline_total_error: float
    enhanced_total_error: float

    # Betting performance (vs the line)
    baseline_spread_correct: Optional[bool]  # Beat the spread
    enhanced_spread_correct: Optional[bool]
    baseline_total_correct: Optional[bool]   # Over/under correct
    enhanced_total_correct: Optional[bool]


@dataclass
class BacktestSummary:
    """Summary statistics for backtest results."""
    total_games: int

    # Spread accuracy
    baseline_spread_mae: float  # Mean Absolute Error
    enhanced_spread_mae: float
    spread_improvement: float  # Percentage improvement

    # Total accuracy
    baseline_total_mae: float
    enhanced_total_mae: float
    total_improvement: float

    # Betting performance (vs the line)
    baseline_spread_win_pct: Optional[float]
    enhanced_spread_win_pct: Optional[float]
    baseline_total_win_pct: Optional[float]
    enhanced_total_win_pct: Optional[float]

    # Confidence intervals
    spread_improvement_significant: bool
    total_improvement_significant: bool


class GamePredictionsBacktest:
    """
    Backtest game predictions to validate metrics improvements.

    Compares baseline (simple PPG) vs enhanced (pace/turnover/efficiency)
    predictions against actual game results.
    """

    def __init__(self, season: int = 2024, inputs_dir: str = "inputs"):
        """
        Initialize backtest engine.

        Args:
            season: Season to backtest
            inputs_dir: Directory with input data
        """
        self.season = season
        self.inputs_dir = inputs_dir

        # Load historical games
        self.games_df = self._load_games()

        # Initialize analyzers
        self.baseline_analyzer = GameMarketAnalyzer(
            season=season,
            use_enhanced_metrics=False
        )

        self.enhanced_analyzer = GameMarketAnalyzer(
            season=season,
            use_enhanced_metrics=True
        )

    def _load_games(self) -> pd.DataFrame:
        """Load historical games with results and betting lines."""
        schedule_file = f"{self.inputs_dir}/schedules_2024_2025.csv"

        df = pd.read_csv(schedule_file)

        # Filter to regular season games for this season
        df = df[
            (df['season'] == self.season) &
            (df['game_type'] == 'REG') &
            (df['away_score'].notna()) &  # Game completed
            (df['home_score'].notna())
        ].copy()

        # Calculate actual spread (positive = home won)
        df['actual_spread'] = df['home_score'] - df['away_score']
        df['actual_total'] = df['home_score'] + df['away_score']

        return df

    def run_backtest(
        self,
        weeks: Optional[List[int]] = None,
        recent_weeks: int = 4
    ) -> List[PredictionResult]:
        """
        Run backtest on specified weeks.

        Args:
            weeks: List of weeks to test (default: all completed weeks)
            recent_weeks: Number of recent weeks to use for metrics

        Returns:
            List of prediction results
        """
        if weeks is None:
            weeks = sorted(self.games_df['week'].unique())

        results = []

        for week in weeks:
            week_games = self.games_df[self.games_df['week'] == week]

            if len(week_games) == 0:
                continue

            print(f"\nBacktesting Week {week} ({len(week_games)} games)...")

            for _, game in week_games.iterrows():
                try:
                    result = self._backtest_game(game, recent_weeks)
                    results.append(result)
                except Exception as e:
                    print(f"  ⚠️  Skipping {game['away_team']}@{game['home_team']}: {e}")
                    continue

        return results

    def _backtest_game(
        self,
        game: pd.Series,
        recent_weeks: int
    ) -> PredictionResult:
        """Backtest a single game."""
        home_team = game['home_team']
        away_team = game['away_team']
        week = int(game['week'])

        # Generate baseline prediction (no enhanced metrics)
        baseline_pred = self.baseline_analyzer.predict_game_outcome(
            home_team=home_team,
            away_team=away_team,
            week=week,
            recent_weeks=recent_weeks
        )

        # Generate enhanced prediction (with pace/turnover/efficiency)
        enhanced_pred = self.enhanced_analyzer.predict_game_outcome(
            home_team=home_team,
            away_team=away_team,
            week=week,
            recent_weeks=recent_weeks
        )

        # Actual results
        actual_home = float(game['home_score'])
        actual_away = float(game['away_score'])
        actual_spread = actual_home - actual_away
        actual_total = actual_home + actual_away

        # Betting lines (may be NaN)
        line_spread = game.get('spread_line')
        if pd.isna(line_spread):
            line_spread = None
        else:
            line_spread = float(line_spread)

        line_total = game.get('total_line')
        if pd.isna(line_total):
            line_total = None
        else:
            line_total = float(line_total)

        # Calculate errors
        baseline_spread_error = abs(baseline_pred.predicted_spread - actual_spread)
        enhanced_spread_error = abs(enhanced_pred.predicted_spread - actual_spread)
        baseline_total_error = abs(baseline_pred.predicted_total - actual_total)
        enhanced_total_error = abs(enhanced_pred.predicted_total - actual_total)

        # Betting performance (vs the line)
        baseline_spread_correct = None
        enhanced_spread_correct = None
        baseline_total_correct = None
        enhanced_total_correct = None

        if line_spread is not None:
            # Did prediction beat the spread?
            # If line is +3 (home favored), home needs to win by >3 to cover
            baseline_spread_correct = (
                (baseline_pred.predicted_spread > line_spread and actual_spread > line_spread) or
                (baseline_pred.predicted_spread < line_spread and actual_spread < line_spread)
            )
            enhanced_spread_correct = (
                (enhanced_pred.predicted_spread > line_spread and actual_spread > line_spread) or
                (enhanced_pred.predicted_spread < line_spread and actual_spread < line_spread)
            )

        if line_total is not None:
            # Did prediction get over/under correct?
            baseline_total_correct = (
                (baseline_pred.predicted_total > line_total and actual_total > line_total) or
                (baseline_pred.predicted_total < line_total and actual_total < line_total)
            )
            enhanced_total_correct = (
                (enhanced_pred.predicted_total > line_total and actual_total > line_total) or
                (enhanced_pred.predicted_total < line_total and actual_total < line_total)
            )

        return PredictionResult(
            game_id=game['game_id'],
            week=week,
            home_team=home_team,
            away_team=away_team,
            actual_home_score=actual_home,
            actual_away_score=actual_away,
            actual_spread=actual_spread,
            actual_total=actual_total,
            line_spread=line_spread,
            line_total=line_total,
            baseline_home_score=baseline_pred.home_score,
            baseline_away_score=baseline_pred.away_score,
            baseline_spread=baseline_pred.predicted_spread,
            baseline_total=baseline_pred.predicted_total,
            enhanced_home_score=enhanced_pred.home_score,
            enhanced_away_score=enhanced_pred.away_score,
            enhanced_spread=enhanced_pred.predicted_spread,
            enhanced_total=enhanced_pred.predicted_total,
            baseline_spread_error=baseline_spread_error,
            enhanced_spread_error=enhanced_spread_error,
            baseline_total_error=baseline_total_error,
            enhanced_total_error=enhanced_total_error,
            baseline_spread_correct=baseline_spread_correct,
            enhanced_spread_correct=enhanced_spread_correct,
            baseline_total_correct=baseline_total_correct,
            enhanced_total_correct=enhanced_total_correct
        )

    def calculate_summary(self, results: List[PredictionResult]) -> BacktestSummary:
        """Calculate summary statistics from backtest results."""
        if not results:
            raise ValueError("No results to summarize")

        # Spread accuracy
        baseline_spread_errors = [r.baseline_spread_error for r in results]
        enhanced_spread_errors = [r.enhanced_spread_error for r in results]

        baseline_spread_mae = np.mean(baseline_spread_errors)
        enhanced_spread_mae = np.mean(enhanced_spread_errors)
        spread_improvement = (baseline_spread_mae - enhanced_spread_mae) / baseline_spread_mae * 100

        # Total accuracy
        baseline_total_errors = [r.baseline_total_error for r in results]
        enhanced_total_errors = [r.enhanced_total_error for r in results]

        baseline_total_mae = np.mean(baseline_total_errors)
        enhanced_total_mae = np.mean(enhanced_total_errors)
        total_improvement = (baseline_total_mae - enhanced_total_mae) / baseline_total_mae * 100

        # Betting performance (vs the line)
        spread_results = [r for r in results if r.baseline_spread_correct is not None]
        total_results = [r for r in results if r.baseline_total_correct is not None]

        baseline_spread_win_pct = None
        enhanced_spread_win_pct = None
        if spread_results:
            baseline_spread_win_pct = sum(1 for r in spread_results if r.baseline_spread_correct) / len(spread_results)
            enhanced_spread_win_pct = sum(1 for r in spread_results if r.enhanced_spread_correct) / len(spread_results)

        baseline_total_win_pct = None
        enhanced_total_win_pct = None
        if total_results:
            baseline_total_win_pct = sum(1 for r in total_results if r.baseline_total_correct) / len(total_results)
            enhanced_total_win_pct = sum(1 for r in total_results if r.enhanced_total_correct) / len(total_results)

        # Statistical significance (simple t-test approximation)
        # If improvement > 2 standard errors, consider significant
        spread_std = np.std([e - b for e, b in zip(enhanced_spread_errors, baseline_spread_errors)])
        total_std = np.std([e - b for e, b in zip(enhanced_total_errors, baseline_total_errors)])

        spread_improvement_significant = abs(baseline_spread_mae - enhanced_spread_mae) > 2 * spread_std / np.sqrt(len(results))
        total_improvement_significant = abs(baseline_total_mae - enhanced_total_mae) > 2 * total_std / np.sqrt(len(results))

        return BacktestSummary(
            total_games=len(results),
            baseline_spread_mae=baseline_spread_mae,
            enhanced_spread_mae=enhanced_spread_mae,
            spread_improvement=spread_improvement,
            baseline_total_mae=baseline_total_mae,
            enhanced_total_mae=enhanced_total_mae,
            total_improvement=total_improvement,
            baseline_spread_win_pct=baseline_spread_win_pct,
            enhanced_spread_win_pct=enhanced_spread_win_pct,
            baseline_total_win_pct=baseline_total_win_pct,
            enhanced_total_win_pct=enhanced_total_win_pct,
            spread_improvement_significant=spread_improvement_significant,
            total_improvement_significant=total_improvement_significant
        )

    def print_report(self, results: List[PredictionResult]):
        """Print detailed backtest report."""
        if not results:
            print("No results to report")
            return

        summary = self.calculate_summary(results)

        print("\n" + "="*70)
        print("GAME PREDICTIONS BACKTEST REPORT")
        print("="*70)

        print(f"\nSeason: {self.season}")
        print(f"Games Tested: {summary.total_games}")
        print(f"Weeks: {min(r.week for r in results)}-{max(r.week for r in results)}")

        # Spread Accuracy
        print("\n" + "-"*70)
        print("SPREAD PREDICTION ACCURACY")
        print("-"*70)
        print(f"Baseline MAE:      {summary.baseline_spread_mae:.2f} points")
        print(f"Enhanced MAE:      {summary.enhanced_spread_mae:.2f} points")
        print(f"Improvement:       {summary.spread_improvement:+.1f}%")

        if summary.spread_improvement_significant:
            print("Statistical Significance: ✓ SIGNIFICANT (p < 0.05)")
        else:
            print("Statistical Significance: ⚠️  Not significant")

        # Total Accuracy
        print("\n" + "-"*70)
        print("TOTAL PREDICTION ACCURACY")
        print("-"*70)
        print(f"Baseline MAE:      {summary.baseline_total_mae:.2f} points")
        print(f"Enhanced MAE:      {summary.enhanced_total_mae:.2f} points")
        print(f"Improvement:       {summary.total_improvement:+.1f}%")

        if summary.total_improvement_significant:
            print("Statistical Significance: ✓ SIGNIFICANT (p < 0.05)")
        else:
            print("Statistical Significance: ⚠️  Not significant")

        # Betting Performance
        if summary.baseline_spread_win_pct is not None:
            print("\n" + "-"*70)
            print("BETTING PERFORMANCE (vs Vegas Lines)")
            print("-"*70)
            print(f"\nSpread Betting (ATS):")
            print(f"  Baseline Win %:    {summary.baseline_spread_win_pct:.1%}")
            print(f"  Enhanced Win %:    {summary.enhanced_spread_win_pct:.1%}")
            print(f"  Improvement:       {(summary.enhanced_spread_win_pct - summary.baseline_spread_win_pct)*100:+.1f}%")

            # 52.4% is breakeven with -110 odds
            if summary.enhanced_spread_win_pct >= 0.524:
                print(f"  ROI Status:        ✓ PROFITABLE (>52.4% breakeven)")
            else:
                print(f"  ROI Status:        ⚠️  Below breakeven (need 52.4%)")

        if summary.baseline_total_win_pct is not None:
            print(f"\nTotal Betting (O/U):")
            print(f"  Baseline Win %:    {summary.baseline_total_win_pct:.1%}")
            print(f"  Enhanced Win %:    {summary.enhanced_total_win_pct:.1%}")
            print(f"  Improvement:       {(summary.enhanced_total_win_pct - summary.baseline_total_win_pct)*100:+.1f}%")

            if summary.enhanced_total_win_pct >= 0.524:
                print(f"  ROI Status:        ✓ PROFITABLE (>52.4% breakeven)")
            else:
                print(f"  ROI Status:        ⚠️  Below breakeven (need 52.4%)")

        # Sample predictions
        print("\n" + "-"*70)
        print("SAMPLE PREDICTIONS (First 5 Games)")
        print("-"*70)

        for i, result in enumerate(results[:5]):
            print(f"\nGame {i+1}: {result.away_team} @ {result.home_team} (Week {result.week})")
            print(f"  Actual:    {result.away_team} {result.actual_away_score:.0f} - {result.home_team} {result.actual_home_score:.0f}")
            print(f"  Baseline:  {result.away_team} {result.baseline_away_score:.1f} - {result.home_team} {result.baseline_home_score:.1f}")
            print(f"  Enhanced:  {result.away_team} {result.enhanced_away_score:.1f} - {result.home_team} {result.enhanced_home_score:.1f}")
            print(f"  Spread Errors: Baseline {result.baseline_spread_error:.1f} pts, Enhanced {result.enhanced_spread_error:.1f} pts")
            print(f"  Total Errors:  Baseline {result.baseline_total_error:.1f} pts, Enhanced {result.enhanced_total_error:.1f} pts")

        print("\n" + "="*70)

        # Overall verdict
        print("\nOVERALL ASSESSMENT:")
        if summary.spread_improvement > 5 and summary.total_improvement > 5:
            print("✅ Enhanced metrics show STRONG improvement over baseline")
        elif summary.spread_improvement > 0 and summary.total_improvement > 0:
            print("✅ Enhanced metrics show modest improvement over baseline")
        elif summary.spread_improvement < 0 or summary.total_improvement < 0:
            print("⚠️  Enhanced metrics show mixed results vs baseline")
        else:
            print("➖ Enhanced metrics show similar performance to baseline")

        print("\n" + "="*70 + "\n")

    def export_results(self, results: List[PredictionResult], output_file: str):
        """Export results to CSV for detailed analysis."""
        df = pd.DataFrame([
            {
                'game_id': r.game_id,
                'week': r.week,
                'home_team': r.home_team,
                'away_team': r.away_team,
                'actual_home_score': r.actual_home_score,
                'actual_away_score': r.actual_away_score,
                'actual_spread': r.actual_spread,
                'actual_total': r.actual_total,
                'line_spread': r.line_spread,
                'line_total': r.line_total,
                'baseline_spread': r.baseline_spread,
                'enhanced_spread': r.enhanced_spread,
                'baseline_total': r.baseline_total,
                'enhanced_total': r.enhanced_total,
                'baseline_spread_error': r.baseline_spread_error,
                'enhanced_spread_error': r.enhanced_spread_error,
                'baseline_total_error': r.baseline_total_error,
                'enhanced_total_error': r.enhanced_total_error,
                'baseline_spread_correct': r.baseline_spread_correct,
                'enhanced_spread_correct': r.enhanced_spread_correct,
                'baseline_total_correct': r.baseline_total_correct,
                'enhanced_total_correct': r.enhanced_total_correct
            }
            for r in results
        ])

        df.to_csv(output_file, index=False)
        print(f"Results exported to {output_file}")
